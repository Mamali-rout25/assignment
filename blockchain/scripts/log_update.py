import os
import json
from hashlib import sha256
from web3 import Web3


def compute_hash(model_bytes: bytes) -> str:
    return sha256(model_bytes).hexdigest()


def main():
    provider = os.environ.get("WEB3_PROVIDER", "http://127.0.0.1:8545")
    contract_info_path = os.environ.get("CONTRACT_INFO", "blockchain/contract_info.json")
    if not os.path.exists(contract_info_path):
        raise FileNotFoundError(f"Contract info file not found: {contract_info_path}")

    with open(contract_info_path, "r", encoding="utf-8") as fp:
        contract_info = json.load(fp)

    web3 = Web3(Web3.HTTPProvider(provider))
    if not web3.is_connected():
        raise RuntimeError(f"Could not connect to provider at {provider}")

    contract = web3.eth.contract(address=contract_info["address"], abi=contract_info["abi"])
    account = web3.eth.accounts[0]

    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Set MODEL_PATH environment variable to the model file to hash and log.")

    with open(model_path, "rb") as fp:
        model_bytes = fp.read()

    model_hash = compute_hash(model_bytes)
    dataset = os.environ.get("DATASET", "unknown")
    variant = os.environ.get("VARIANT", "unknown")
    round_num = int(os.environ.get("ROUND", "0"))
    cid = os.environ.get("CID", "")
    model_hash_bytes = bytes.fromhex(model_hash)

    tx = contract.functions.logUpdate(model_hash_bytes, dataset, variant, round_num, cid).transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx)
    print(f"Logged update on chain in tx {receipt.transactionHash.hex()}")


if __name__ == "__main__":
    main()
