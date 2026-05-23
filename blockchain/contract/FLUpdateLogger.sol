// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLUpdateLogger {
    struct UpdateRecord {
        bytes32 modelHash;
        string dataset;
        string variant;
        uint256 round;
        string cid;
        uint256 timestamp;
    }

    UpdateRecord[] public updates;
    event UpdateLogged(bytes32 indexed modelHash, string dataset, string variant, uint256 round, string cid, uint256 timestamp);

    function logUpdate(
        bytes32 modelHash,
        string calldata dataset,
        string calldata variant,
        uint256 round,
        string calldata cid
    ) external {
        updates.push(UpdateRecord({
            modelHash: modelHash,
            dataset: dataset,
            variant: variant,
            round: round,
            cid: cid,
            timestamp: block.timestamp
        }));
        emit UpdateLogged(modelHash, dataset, variant, round, cid, block.timestamp);
    }

    function getUpdateCount() external view returns (uint256) {
        return updates.length;
    }

    function getUpdate(uint256 index) external view returns (bytes32, string memory, string memory, uint256, string memory, uint256) {
        UpdateRecord storage record = updates[index];
        return (record.modelHash, record.dataset, record.variant, record.round, record.cid, record.timestamp);
    }
}
