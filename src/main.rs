use log::{error, info};
use pachamama::api::account::GetAccountResp;

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
