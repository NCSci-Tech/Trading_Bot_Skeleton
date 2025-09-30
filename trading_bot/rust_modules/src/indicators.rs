[package]
name = "indicators"
version = "0.1.0"
edition = "2021"
authors = ["Trading Bot Team"]
description = "High-performance technical indicators for trading bot"
license = "MIT"

[lib]
name = "indicators"
crate-type = ["cdylib"]

[dependencies]
pyo3.workspace = true
numpy.workspace = true
serde.workspace = true
serde_json.workspace = true
rayon.workspace = true
chrono.workspace = true
anyhow.workspace = true
ndarray.workspace = true

[dependencies.pyo3]
workspace = true
features = ["extension-module"]