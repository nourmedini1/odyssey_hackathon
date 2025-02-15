import requests
import numpy as np
from datetime import datetime
import time
import logging
from typing import Dict, List
import dotenv
from mistralai import Mistral
import schedule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEYS = {
    'COINMARKETCAP': dotenv.get_key('.env', 'COINMARKETCAP_API_KEY'),
    'ETHERSCAN': dotenv.get_key('.env', 'ETHERSCAN_API_KEY'),
    'BINANCE_API_KEY': dotenv.get_key('.env', 'BINANCE_API_KEY_1'),
    'BINANCE_API_SECRET': dotenv.get_key('.env', 'BINANCE_SECRET_KEY_1'),
    'MISTRAL_API_KEY': dotenv.get_key('.env', 'MISTRAL_API_KEY')
}

MISTRAL_MODEL_NAME = "mistral-large-latest"

PND_DETECTION_MODEL_ENDPOINT = "http://20.199.80.240:5040/predict"
SENTIMENT_ANALYSIS_MODEL_TELEMEGRAM_MESSAGES_ENDPOINT = "http://20.199.80.240:5030/telegram/messages"
AI_BLOGGER_ENDPOINT = "http://20.199.80.240:5050/send-message"
BLACKLIST = ['USDT', 'USDC', 'DAI']

llm = Mistral(api_key=API_KEYS['MISTRAL_API_KEY'])


def get_valid_binance_pairs() -> set:
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        valid_pairs = {s["symbol"] for s in data["symbols"]}
        return valid_pairs
    except Exception as e:
        logging.error("Error fetching Binance exchange info: " + str(e))
        return set()


def validate_binance_pair(symbol: str, valid_pairs: set) -> str:
    binance_symbol = symbol.upper() + "USDT"
    if binance_symbol in valid_pairs:
        return binance_symbol
    else:
        logging.warning(f"Invalid Binance pair: {binance_symbol}")
        return None


def is_valid_erc20(address: str) -> bool:
    return address is not None and address.startswith("0x") and len(address) == 42


def sanitize_features(features: Dict) -> Dict:
    MAX_VALUES = {
        'std_rush_order': 10,
        'avg_rush_order': 10,
        'std_trades': 100,
        'std_volume': 1e6,
        'avg_volume': 1e6,
        'std_price': 1,
        'avg_price': 1e4,
        'avg_price_max': 1e4
    }
    sanitized = {}
    for key, value in features.items():
        if isinstance(value, (int, float)):
            sanitized[key] = min(value, MAX_VALUES.get(key, float('inf')))
        else:
            sanitized[key] = value
    return sanitized


class PumpDetector:
    def __init__(self):
        self.cmc_headers = {'X-CMC_PRO_API_KEY': API_KEYS['COINMARKETCAP']}
        self.base_params = {
            'convert': 'USD',
            'aux': 'num_market_pairs,date_added,platform'
        }
        self.valid_binance_pairs = get_valid_binance_pairs()

    def get_risky_coins(self) -> List[Dict]:
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            params = {**self.base_params, 'start': 1, 'limit': 500}
            response = requests.get(url, headers=self.cmc_headers, params=params)
            response.raise_for_status()
            data = response.json()
            risky_coins = []
            current_time = datetime.now()
            for coin in data['data']:
                try:
                    if coin['symbol'] in BLACKLIST:
                        continue
                    added_date = datetime.strptime(coin['date_added'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    age_days = (current_time - added_date).days
                    market_cap = coin['quote']['USD']['market_cap']
                    volume = coin['quote']['USD']['volume_24h']
                    if (market_cap < 1e8 or volume < 5e6) or age_days < 365 or coin['num_market_pairs'] < 3:
                        contract_address = None
                        if coin.get('platform'):
                            addr = coin['platform'].get('token_address')
                            if is_valid_erc20(addr):
                                contract_address = addr
                            else:
                                logging.warning(f"Invalid contract address for {coin['symbol']}")
                        risky_coins.append({
                            'symbol': coin['symbol'],
                            'contract_address': contract_address,
                            'market_cap': market_cap,
                            'volume_24h': volume,
                            'age_days': age_days
                        })
                except KeyError as e:
                    logging.warning(f"Missing field {str(e)} in coin data")
                    continue
            logging.info(f"Found {len(risky_coins)} risky coins.")
            return sorted(risky_coins, key=lambda x: x['market_cap'])[:10]
        except requests.exceptions.RequestException as e:
            logging.error(f"CMC API Error: {str(e)}")
            return []

    def get_transaction_features(self, contract_address: str) -> Dict:
        if not contract_address or not is_valid_erc20(contract_address):
            logging.warning("Invalid or missing contract address; skipping Etherscan data.")
            return {'std_rush_order': 0.0, 'avg_rush_order': 0.0}
        try:
            end_time = int(time.time())
            start_time = end_time - 3600
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract_address,
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'asc',
                'apikey': API_KEYS['ETHERSCAN'],
                'startTimestamp': start_time,
                'endTimestamp': end_time
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data['status'] != '1':
                logging.warning(f"Etherscan error: {data.get('message', 'Unknown error')}")
                return {'std_rush_order': 0.0, 'avg_rush_order': 0.0}
            tx_list = data['result']
            if not tx_list:
                return {'std_rush_order': 0.0, 'avg_rush_order': 0.0}
            values = [float(tx['value']) / 1e18 for tx in tx_list]
            timestamps = [int(tx['timeStamp']) for tx in tx_list]
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            time_window = max_ts - min_ts if max_ts > min_ts else 1
            weighted_values = [v * ((ts - min_ts) / time_window) for v, ts in zip(values, timestamps)]
            std_rush_order = np.std(weighted_values, ddof=1) if len(weighted_values) > 1 else 0.0
            avg_rush_order = np.mean(weighted_values)
            return {
                'std_rush_order': std_rush_order,
                'avg_rush_order': avg_rush_order
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"Etherscan request failed: {str(e)}")
            return {'std_rush_order': 0.0, 'avg_rush_order': 0.0}

    def get_market_features(self, symbol: str) -> Dict:
        valid_symbol = validate_binance_pair(symbol, self.valid_binance_pairs)
        if not valid_symbol:
            logging.warning(f"Skipping Binance market features for invalid symbol: {symbol}USDT")
            return {}
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - 3600000
            trades = []
            while True:
                url = "https://api.binance.com/api/v3/aggTrades"
                params = {
                    'symbol': valid_symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                new_trades = response.json()
                if not new_trades:
                    break
                trades.extend(new_trades)
                start_time = int(new_trades[-1]['T']) + 1
                if len(new_trades) < 1000:
                    break
                time.sleep(0.2)
            if not trades:
                return {}
            prices = [float(t['p']) for t in trades]
            volumes = [float(t['q']) for t in trades]
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))] if len(prices) > 1 else []
            std_trades = np.std([1 if t['m'] else -1 for t in trades], ddof=1) if len(trades) > 1 else 0.0
            std_volume = np.std(volumes, ddof=1) if len(volumes) > 1 else 0.0
            avg_volume = np.mean(volumes) if volumes else 0.0
            std_price = np.std(price_changes, ddof=1) if len(price_changes) > 1 else 0.0
            avg_price = np.mean(prices) if prices else 0.0
            avg_price_max = max(prices) if prices else 0.0
            return {
                'std_trades': std_trades,
                'std_volume': std_volume,
                'avg_volume': avg_volume,
                'std_price': std_price,
                'avg_price': avg_price,
                'avg_price_max': avg_price_max
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"Binance API Error: {str(e)}")
            return {}

    def calculate_time_features(self) -> Dict:
        now = datetime.now()
        return {
            'hour_sin': np.sin(2 * np.pi * now.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24),
            'minute_sin': np.sin(2 * np.pi * now.minute / 60),
            'minute_cos': np.cos(2 * np.pi * now.minute / 60)
        }

    def generate_features(self, symbol: str, contract_address: str) -> Dict:
        features = {}
        features.update(self.calculate_time_features())
        features.update(self.get_transaction_features(contract_address))
        features.update(self.get_market_features(symbol))
        expected_keys = [
            'std_rush_order', 'avg_rush_order',
            'std_trades', 'std_volume', 'avg_volume',
            'std_price', 'avg_price', 'avg_price_max',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'
        ]
        for key in expected_keys:
            if key not in features or features[key] is None:
                features[key] = 0.0
        return sanitize_features(features)

    def analyze_coins(self):
        risky_coins = self.get_risky_coins()
        if not risky_coins:
            logging.warning("No risky coins found")
            return []
        predictions = []
        expected_keys = {
            "std_rush_order", "avg_rush_order",
            "std_trades", "std_volume", "avg_volume",
            "std_price", "avg_price", "avg_price_max",
            "hour_sin", "hour_cos", "minute_sin", "minute_cos"
        }
        for coin in risky_coins:
            try:
                logging.info(f"Processing {coin['symbol']}")
                features = self.generate_features(coin['symbol'], coin['contract_address'])
                model_payload = { key: features.get(key, 0.0) for key in expected_keys }
                logging.info("Payload to model: %s", model_payload)
                headers = {"Content-Type": "application/json"}
                response = requests.post(PND_DETECTION_MODEL_ENDPOINT, json=model_payload, headers=headers, timeout=20)
                response.raise_for_status()
                prediction = response.json()
                predictions.append({
                    'symbol': coin['symbol'],
                    'features': features,
                    'prediction': prediction
                })
                time.sleep(1)
            except Exception as e:
                logging.error(f"Failed processing {coin['symbol']}: {str(e)}")
                continue
        return predictions


def get_sentiment_analysis():
    try:
        response = requests.get(SENTIMENT_ANALYSIS_MODEL_TELEMEGRAM_MESSAGES_ENDPOINT)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sentiment analysis data: {str(e)}")
        return {}


def run_cron_job():
    logging.info("Starting cron job...")
    detector = PumpDetector()
    results = detector.analyze_coins()
    sentiment = get_sentiment_analysis()
    logging.info("Cron job finished. Results: %s", results)
    logging.info("Sentiment analysis: %s", sentiment)
    analysis = sentiment.get("analysis")
    is_pump_and_dump = analysis.get("is_pump_and_dump", False)
    sentiment_summary = analysis.get("summary", "No sentiment summary provided.")
    if is_pump_and_dump or any(result.get("prediction") == 1 for result in results):
        prompt = f"""
You are an expert crypto blogger. Based on the current market conditions, here are the details:
- Pump & Dump Detected: {is_pump_and_dump}
- Sentiment Analysis Summary: {sentiment_summary}
- Risky Coin Predictions: {results}

Write a short and concize engaging, insightful blog about it , make sure you make it shott
"""
        logging.info("Generated prompt for Mistral: %s", prompt)
        
        try:
            llm_response = llm.chat.complete(
                model=MISTRAL_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            blog_text = llm_response.choices[0].message.content
            logging.info("Generated blog text: %s", blog_text)
        except Exception as e:
            logging.error(f"Failed to generate blog text: {e}")
            return
        
        try:
            blogger_response = requests.post(AI_BLOGGER_ENDPOINT, json={"blog_text": blog_text}, timeout=20)
            blogger_response.raise_for_status()
            logging.info("Blog sent successfully via blogger API.")
        except Exception as e:
            logging.error(f"Failed to send blog via blogger API: {e}")
    else:
        logging.info("No pump and dump indicators found and no risky predictions; skipping blog generation.")


if __name__ == "__main__":
    run_cron_job()
    schedule.every(4).hours.do(run_cron_job)
    logging.info("Cron scheduler started. Waiting for the next scheduled run...")
    while True:
        schedule.run_pending()
        time.sleep(60)
