<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/logos/UCM.png">
    <img alt="UCM" src="docs/source/logos/UCM-light.png" width=50%>
  </picture>
</p>

<p align="center">
| <a href="docs/source/index.md"><b>Documentation</b></a> | <a href="https://github.com/ModelEngine-Group/unified-cache-management/issues/16"><b>Roadmap</b></a> |
</p>

---

*Latest News* 🔥
- [2025/08/01] We are excited to announce the alpha release of Unified Cache Manager.

---

## Performance
nfs connector has reached about 4x TTFT accelerate.

![perf](docs/source/images/nfs_performance.png)

## Overview

### Motivation
With the increase of model size, the KV cache became larger and sparser, especially for long sequence requests. To reduce the GPU memory used, offload full KV to external storage and only keep partial or compressed KV in GPU memory became the popular direction. This can also reduce the GPU calculation, increase the sequence length and batch size of decoding.

Sparse KV cache have many different choices. Recently paper point out that there is no common way can fit all scenarios and all models. So better to build a common framework then different sparse algorithms can be plugin to it like KV connector for PC.

### Proposed Change
![idea](docs/source/images/idea.png)

All gray boxes are current classes in 0.9.2. Green boxes are proposed to add. Light green ones show out the future sub classes base on this framework.

SpareKVBase is the base class of different algorithms. Just like KV connector design, it will hook few places of scheduler and layer.py to allow sparse algorithms do additional load, dump and calculate sparse KV blocks.

SparseKVManager provide different KV block allocation methods for different algorithms. To keep all implementation under SpareKVBase, it will call SparseKVBase and real implementation will happen in sub class of sparse algorithms.

KVStoreBase helps decoupling sparse algorithms and external storage. It defined the methods how to talk to external storage, so any sparse algorithms can work with any external storage. Concepts here is blocks identify by ID with offset. This is not only for sparse but also naturally for prefix cache also. KVStoreConnector connect it with current KVConnectorBase_V1 to provide PC function.

NFSStore is sample implementation here provide ability to store blocks in local file system or NFS mount point in multi-server case.

LocalCachedStore can reference any store to provide local DRAM read cache layer.

---

## Quick Start
please refer to [installation](docs/source/getting-started/installation.md) and [example](docs/source/getting-started/example/dram_conn.md)。

---

## Support Features
please refer to [features matrix](docs/source/feature/support.md).

---

## Branch Policy
Unified Cache has main branch, develop branch and release branch.
- **main**: The main branch is the stable line; it receives merges from develop once basic tests pass. In principle, there should be no direct check-ins to the main branch.
- **develop**: The develop branch is the daily development branch where new features are merged.
- **release**: Each time a new version release process begins, release branch is merged from the main branch. This branch only accepts bug fixes, and all bug fixes will be picked back to the develop branch. After the release testing passes, a release tag (x.x.x) will be created on the release branch. The release branch will be retained for the next release.

---

## Contributing
When you want to contribute some features to the Unified Cache Community, first fork a branch (usually develop) to your own repository, then commit in your own repository, and finally submit a pull request to the community.

---

## License

UCM is licensed under the MIT with additional conditions. Please read the [LICENSE](./LICENSE) file for details.
