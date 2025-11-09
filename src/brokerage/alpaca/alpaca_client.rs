//! # Alpaca Client
//!
//! Currently implements the ["legacy" authentication flow](https://docs.alpaca.markets/docs/authentication) which embeds the Key ID and Secret Key as part of the headers in the REST API request.
//!
//! Check [Alpaca's status here](https://status.alpaca.markets/)
use crate::brokerage::alpaca::api::account;
use crate::brokerage::alpaca::api::API_VERSION;
use crate::brokerage::alpaca::{
    ENV_API_KEY_ID, ENV_API_SECRET_KEY_ID, LIVE_ACCT_API_ENDPOINT, PAPER_ACCT_API_ENDPOINT,
};
use anyhow::{anyhow, ensure, Context};
use log::{debug, info, warn};
use std::sync::Arc;

/// Account type to use
pub enum AccountType {
    /// Paper (non-real money) account
    PAPER,
    /// Live (real money) account
    LIVE,
}

/// Alpaca Markets client
pub struct AlpacaClient {
    /// APCA-API-KEY-ID
    api_key_id: Arc<String>,
    /// APCA-API-SECRET-KEY
    api_secret_key: Arc<String>,
    /// Base URL for trading API requests
    api_base_url: Arc<String>,
    /// [reqwest client](https://docs.rs/reqwest/latest/reqwest/struct.Client.html) to be reused
    /// for requests
    reqwest_client: reqwest::Client,
}

impl AlpacaClient {
    pub fn new(acct_type: AccountType) -> anyhow::Result<Self> {
        // Get environment vars for auth
        let key_id = std::env::var(ENV_API_KEY_ID)
            .with_context(|| "'API_KEY_ID' environment variable is not set")?;
        let secret_key = std::env::var(ENV_API_SECRET_KEY_ID)
            .with_context(|| "'API_SECRET_KEY_ID' environment variable is not set")?;

        let api_url = match acct_type {
            AccountType::PAPER => format!("{PAPER_ACCT_API_ENDPOINT}/{API_VERSION}/"),
            AccountType::LIVE => format!("{LIVE_ACCT_API_ENDPOINT}/{API_VERSION}/"),
        };
        debug!("Using Alapaca API base URL of: {api_url}");

        let reqwest_client = reqwest::Client::new();
        debug!("New alpaca client created");

        Ok(AlpacaClient {
            api_key_id: Arc::new(key_id),
            api_secret_key: Arc::new(secret_key),
            api_base_url: Arc::new(api_url),
            reqwest_client,
        })
    }

    /// Get account details and verify account is active and unblocked from trading.
    pub async fn get_account_details(&self) -> anyhow::Result<()> {
        let req_url = self.api_base_url.clone().to_string() + account::ENDPOINT;
        let response = self
            .reqwest_client
            .get(req_url)
            .header("APCA-API-KEY-ID", self.api_key_id.clone().to_string())
            .header(
                "APCA-API-SECRET-KEY",
                self.api_secret_key.clone().to_string(),
            )
            .send()
            .await?;

        if response.status() != reqwest::StatusCode::OK {
            warn!("Non-ok GET response");
            Err(anyhow!("Bad request"))
        } else {
            let json_body: account::GetAccountResp = response.json().await?;
            ensure!(json_body.status == "ACTIVE", "Account is not active!");
            if let Some(trade_blocked) = json_body.trade_suspended_by_user {
                ensure!(!trade_blocked, "Trading suspended by user!");
            }
            if let Some(trade_blocked) = json_body.trading_blocked {
                ensure!(!trade_blocked, "Trading currently blocked!");
            }
            if let Some(transfers_blocked) = json_body.transfers_blocked {
                ensure!(!transfers_blocked, "Transfers currently blocked!");
            }
            if let Some(acct_blocked) = json_body.account_blocked {
                ensure!(!acct_blocked, "Account is currently blocked!");
            }
            info!(
                "Current {} portfolio value: ${}, with buying power: ${}",
                json_body.currency.unwrap(),
                json_body.portfolio_value.unwrap(),
                json_body.buying_power.unwrap(),
            );
            Ok(())
        }
    }
}
