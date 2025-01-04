use log::{error, info};
use serde::{Deserialize, Serialize};

/// Response for getting account details, [more info](https://docs.alpaca.markets/reference/getaccount-1)
#[derive(Clone, Deserialize, Debug, Eq, PartialEq, Serialize)]
#[allow(dead_code)]
pub struct GetAccountResp {
    id: String,
    account_number: Option<String>,
    status: String,
    currency: Option<String>,
    cash: Option<String>,
    portfolio_value: Option<String>,
    non_marginable_buying_power: Option<String>,
    accrued_fees: Option<String>,
    pending_transfer_in: Option<String>,
    pending_transfer_out: Option<String>,
    pattern_day_trader: Option<bool>,
    trade_suspended_by_user: Option<bool>,
    trading_blocked: Option<bool>,
    transfers_blocked: Option<bool>,
    account_blocked: Option<bool>,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
    shorting_enabled: Option<bool>,
    long_market_value: Option<String>,
    short_market_value: Option<String>,
    equity: Option<String>,
    last_equity: Option<String>,
    multiplier: Option<String>,
    buying_power: Option<String>,
    initial_margin: Option<String>,
    maintenance_margin: Option<String>,
    sma: Option<String>,
    daytrade_count: Option<i32>,
    balance_asof: Option<String>,
    last_maintenance_margin: Option<String>,
    daytrading_buying_power: Option<String>,
    regt_buying_power: Option<String>,
    options_buying_power: Option<String>,
    options_approved_level: Option<i32>,
    options_trading_level: Option<i32>,
    intraday_adjustments: Option<String>,
    pending_reg_taf_fees: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    info!("Starting connection");

    const PAPER_ACCT_URL: &str = "https://paper-api.alpaca.markets/v2/account";

    const API_KEY_ID: &str = "APCA_API_KEY_ID";
    const API_SECRET_KEY_ID: &str = "APCA_API_SECRET_KEY";

    let key_id = std::env::var(API_KEY_ID)?;
    let secret_key = std::env::var(API_SECRET_KEY_ID)?;

    let response = reqwest::blocking::Client::new()
        .get(PAPER_ACCT_URL)
        .header("APCA-API-KEY-ID", key_id)
        .header("APCA-API-SECRET-KEY", secret_key)
        .send()?;

    if response.status() != reqwest::StatusCode::OK {
        error!("Non-OK HTTP response!");
        return Err("Bad request".into());
    } else {
        let json_body: GetAccountResp = response.json()?;
        dbg!(json_body);
    }

    Ok(())
}
