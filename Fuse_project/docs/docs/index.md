# Four_bar_linkage documentation!

## Description

It is assignment project for fusemachine AI

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://someet-s3-bucket/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://someet-s3-bucket/data/` to `data/`.


