import json
import os
from web3 import Web3
from solcx import compile_standard, install_solc

CONTRACT_PATH = os.path.join(os.path.dirname(__file__), "..", "contract", "FLUpdateLogger.sol")


def compile_contract():
    install_solc("0.8.19")
    with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
        source = file.read()
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"FLUpdateLogger.sol": {"content": source}},
            "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}}},
        },
        solc_version="0.8.19",
    )
    contract_data = compiled["contracts"]["FLUpdateLogger.sol"]["FLUpdateLogger"]
    return contract_data["abi"], contract_data["evm"]["bytecode"]["object"]


def main():
    provider = os.environ.get("WEB3_PROVIDER", "http://127.0.0.1:8545")
    web3 = Web3(Web3.HTTPProvider(provider))
    if not web3.is_connected():
        raise RuntimeError(f"Could not connect to provider at {provider}")

    abi, bytecode = compile_contract()
    account = web3.eth.accounts[0]
    contract = web3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = contract.constructor().transact({"from": account})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    address = tx_receipt.contractAddress
    contract_info = {"address": address, "abi": abi}
    contract_info_path = os.path.join(os.path.dirname(__file__), "..", "contract_info.json")
    with open(contract_info_path, "w", encoding="utf-8") as fp:
        json.dump(contract_info, fp, indent=2)
    print(f"Deployed contract at {address}")
    print(f"Saved contract info to {contract_info_path}")


if __name__ == "__main__":
    main()
