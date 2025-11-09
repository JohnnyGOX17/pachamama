//! # Implementation of Alpaca API and helper methods
//!
//! ## References/Links
//! - [Alpaca Markets Docs](https://docs.alpaca.markets/)
//!   + [Alpaca Learn](https://alpaca.markets/learn)
//!   + [The Options Wheel Strategy Explained (and How to Implement Using Python and Alpaca's Trading API)](https://alpaca.markets/learn/options-wheel-strategy)
//! - [alpaca.markets Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
//! - [d-e-s-o/apca](https://github.com/d-e-s-o/apca): A crate for interacting with the Alpaca API at alpaca.markets.
pub mod alpaca_client;
pub mod api;

const PAPER_ACCT_API_ENDPOINT: &str = "https://paper-api.alpaca.markets";
const LIVE_ACCT_API_ENDPOINT: &str = "https://api.alpaca.markets";
#[allow(unused)]
const MARKET_DATA_ENDPOINT: &str = "https://data.alpaca.markets/";

const ENV_API_KEY_ID: &str = "APCA_API_KEY_ID";
const ENV_API_SECRET_KEY_ID: &str = "APCA_API_SECRET_KEY";
