# Retail Trading Software System - Architecture Specification

## Overview
This specification outlines the complete architecture for a retail trading system designed for a single developer/trader. The system focuses on aggregating diverse data sources, running parallel algorithmic analysis, and surfacing high-potential trading opportunities rather than optimizing for high-frequency trading speed.

**Technology Stack**: Rust (backend), PostgreSQL with TimescaleDB (database), Tokio + Rayon (concurrency), Alpaca Markets (brokerage), React/Next.js (frontend)

---

## 1. Data Sources Integration

### 1.1 Market Data APIs
The following data sources provide comprehensive market information with free or affordable tiers:

**Primary Stock Price Data:**
- [ ] **Alpha Vantage** - Free tier with real-time and historical stock prices, 60+ technical indicators, market news with sentiment analysis
  - Implementation: Create `AlphaVantageClient` module
  - Store API key in configuration
  - Rate limiting: 25 requests/day (free tier) or 75/minute (premium)
  - Endpoints: TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, TECHNICAL_INDICATORS, NEWS_SENTIMENT

- [ ] **Finnhub** - Free real-time stock prices, company fundamentals, economic data
  - Implementation: Create `FinnhubClient` module  
  - Free tier: 60 API calls/minute
  - Endpoints: Quote, Candles, Company Profile, Earnings Calendar, News

- [ ] **Twelve Data** - Stock, forex, crypto market data with technical indicators
  - Implementation: Create `TwelveDataClient` module
  - Free tier: 800 requests/day
  - Endpoints: Time Series, Technical Indicators, Market Movers

- [ ] **Marketstack** - End-of-day and intraday stock data for 70+ exchanges
  - Implementation: Create `MarketstackClient` module
  - Free tier: 100 requests/month for EOD data
  - Best for: Historical backtesting data

- [ ] **Financial Modeling Prep (FMP)** - Comprehensive financial statements, SEC filings, analyst ratings
  - Implementation: Create `FmpClient` module
  - Free tier: 250 requests/day
  - Endpoints: Financial Statements (10-K, 10-Q), Earnings Transcripts, Analyst Estimates, Price Targets

**Alternative Data Sources:**
- [ ] **Polygon.io/Massive.com** - Real-time tick data, comprehensive historical data
  - Implementation: Create `PolygonClient` module
  - Free tier: Limited to previous day's data
  - Endpoints: Aggregates (bars), Trades, Quotes, Tickers

### 1.2 News and Sentiment Data
- [ ] **Marketaux** - Global financial news from 5,000+ sources with sentiment analysis
  - Implementation: Create `MarketauxClient` module
  - Free tier: 100% free with no payment details required
  - Features: NLP sentiment scoring per ticker, entity extraction
  - Endpoints: News feed with entity linking, sentiment scores

- [ ] **Alpha Vantage News API** - Integrated news with market sentiment
  - Already covered in section 1.1
  - Features: AI-powered sentiment analysis, topic extraction

- [ ] **EODHD Financial News** - Stock market news with sentiment analysis
  - Implementation: Create `EodhdClient` module
  - Features: News aggregation, word weights analysis, sentiment scoring
  - Endpoints: News Feed, News Word Weights, Sentiment

### 1.3 Economic and Fundamental Data
- [ ] **Alpha Vantage Economic Indicators** - GDP, unemployment, inflation, interest rates
  - Use existing `AlphaVantageClient`
  - Endpoints: Real GDP, CPI, Federal Funds Rate, Treasury Yields

- [ ] **FMP Company Fundamentals** - Balance sheets, income statements, cash flow
  - Use existing `FmpClient`
  - Endpoints: Income Statement, Balance Sheet, Cash Flow Statement, Financial Ratios

- [ ] **SEC EDGAR Filings** - Direct access to 10-K, 10-Q, 8-K filings
  - Implementation: Create `SecEdgarClient` module
  - Completely free, no API key required
  - Features: Full-text search, CIK lookup, filing history

### 1.4 Options Data
- [ ] **Alpaca Markets Options** - Options chains, Greeks, real-time options quotes
  - Implementation: Create `AlpacaOptionsClient` module
  - Included with Alpaca trading account
  - Endpoints: Options Chains, Options Bars, Latest Options Quote

### 1.5 Additional Alternative Data Sources
- [ ] **Yahoo Finance (via unofficial API)** - Backup source for price data and basic fundamentals
  - Implementation: Create `YahooFinanceClient` module
  - No API key required but use respectfully
  - Note: Unofficial, may have reliability issues

---

## 2. Data Ingest and Database Strategy

### 2.1 Database Architecture

**Database Selection: PostgreSQL with TimescaleDB Extension**

TimescaleDB provides time-series optimization on top of PostgreSQL, which is ideal for:
- Stock price history (time-series data)
- News articles with timestamps
- Economic indicator history
- Trading signals and backtest results

- [ ] Install and configure PostgreSQL 16+ with TimescaleDB extension
- [ ] Create database schema with the following core tables:

```sql
-- Core entities
CREATE TABLE tickers (
    ticker_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Time-series price data (converted to hypertable)
CREATE TABLE stock_prices (
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    adjusted_close DECIMAL(12,4),
    PRIMARY KEY (ticker_id, timestamp)
);
SELECT create_hypertable('stock_prices', 'timestamp');

-- Technical indicators cache
CREATE TABLE technical_indicators (
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    value DECIMAL(20,8),
    metadata JSONB,
    PRIMARY KEY (ticker_id, timestamp, indicator_name)
);
SELECT create_hypertable('technical_indicators', 'timestamp');

-- News articles with full-text search
CREATE TABLE news_articles (
    article_id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    published_at TIMESTAMPTZ NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    sentiment_score DECIMAL(5,4),
    search_vector tsvector,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_news_search ON news_articles USING GIN(search_vector);
CREATE INDEX idx_news_published ON news_articles(published_at DESC);

-- Article-ticker associations (many-to-many)
CREATE TABLE article_tickers (
    article_id INTEGER REFERENCES news_articles(article_id),
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    relevance_score DECIMAL(5,4),
    PRIMARY KEY (article_id, ticker_id)
);

-- Fundamental data
CREATE TABLE fundamentals (
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    fiscal_period DATE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(20,4),
    metadata JSONB,
    PRIMARY KEY (ticker_id, fiscal_period, metric_name)
);

-- Economic indicators
CREATE TABLE economic_indicators (
    indicator_id SERIAL PRIMARY KEY,
    indicator_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DECIMAL(20,8),
    unit VARCHAR(50),
    source VARCHAR(100),
    UNIQUE (indicator_name, timestamp)
);
SELECT create_hypertable('economic_indicators', 'timestamp');

-- Options data
CREATE TABLE options_chains (
    contract_id VARCHAR(50) PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    expiration_date DATE NOT NULL,
    strike_price DECIMAL(12,4) NOT NULL,
    option_type VARCHAR(4) NOT NULL, -- 'call' or 'put'
    last_price DECIMAL(12,4),
    bid DECIMAL(12,4),
    ask DECIMAL(12,4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Data ingestion tracking
CREATE TABLE data_ingestion_log (
    log_id SERIAL PRIMARY KEY,
    source VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    records_processed INTEGER,
    records_inserted INTEGER,
    records_updated INTEGER,
    status VARCHAR(20), -- 'running', 'success', 'failed'
    error_message TEXT,
    metadata JSONB
);
```

### 2.2 Async Data Ingestion Architecture

**Core Pattern**: Use Tokio for I/O-bound API calls, with concurrent connection pooling

- [ ] Create database connection pool using `deadpool-postgres` or `sqlx`
  - Configure pool size: 20-50 connections for concurrent reads/writes
  - Connection lifecycle management with health checks
  - Automatic reconnection on failures

- [ ] Implement `DataIngestionOrchestrator` service:
  ```rust
  pub struct DataIngestionOrchestrator {
      db_pool: Pool<PostgresConnectionManager<NoTls>>,
      api_clients: HashMap<String, Box<dyn DataClient>>,
      rate_limiters: HashMap<String, RateLimiter>,
  }
  ```

- [ ] Create `DataClient` trait for all API integrations:
  ```rust
  #[async_trait]
  pub trait DataClient: Send + Sync {
      async fn fetch_data(&self, request: DataRequest) -> Result<Vec<DataPoint>>;
      fn get_rate_limit(&self) -> RateLimit;
      fn supports_incremental(&self) -> bool;
  }
  ```

- [ ] Implement incremental data ingestion strategy:
  - [ ] Track last successful ingestion timestamp per source per ticker
  - [ ] Query API with `since` or `after` parameters when available
  - [ ] Use database-level upserts (INSERT ... ON CONFLICT) to handle duplicates
  - [ ] Implement change data capture for detecting updates vs. inserts

- [ ] Build rate limiter module using `governor` crate:
  - [ ] Per-API rate limiting with token bucket algorithm
  - [ ] Automatic backoff and retry logic
  - [ ] Queue requests when rate limit is hit
  - [ ] Circuit breaker pattern for API failures

### 2.3 Concurrent Data Pipeline

**Architecture**: Fan-out ingestion with concurrent workers

- [ ] Create ingestion worker pool:
  ```rust
  pub async fn ingest_all_tickers(
      orchestrator: &DataIngestionOrchestrator,
      tickers: Vec<String>,
      sources: Vec<String>,
  ) -> Result<IngestionReport> {
      // Create concurrent streams for each ticker-source combination
      let futures: Vec<_> = tickers
          .iter()
          .cartesian_product(sources.iter())
          .map(|(ticker, source)| {
              orchestrator.ingest_ticker_from_source(ticker, source)
          })
          .collect();
      
      // Execute with controlled concurrency
      let results = futures::stream::iter(futures)
          .buffer_unordered(50) // Max 50 concurrent API calls
          .collect::<Vec<_>>()
          .await;
      
      // Aggregate results
      Ok(IngestionReport::from_results(results))
  }
  ```

- [ ] Implement database write batching:
  - [ ] Collect data points in memory (up to 1000 records or 5 seconds)
  - [ ] Batch INSERT operations using PostgreSQL `COPY` or multi-row INSERT
  - [ ] Use pipelining to overlap network I/O with database writes

- [ ] Create data cleaning and validation pipeline:
  - [ ] Validate data types and ranges before database insertion
  - [ ] Handle missing or null values according to business rules
  - [ ] Detect and log anomalies (e.g., price spikes, volume outliers)
  - [ ] Normalize data formats across different API sources

### 2.4 Scheduled Ingestion Jobs

- [ ] Implement scheduling system using `tokio-cron-scheduler`:
  - [ ] Daily EOD data ingestion (runs after market close at 4:30 PM ET)
  - [ ] Intraday price updates (every 5-15 minutes during market hours)
  - [ ] News ingestion (continuous polling every 10 minutes)
  - [ ] Fundamental data updates (weekly on weekends)
  - [ ] Economic indicators (check for updates daily)

- [ ] Create job configuration system:
  ```rust
  pub struct IngestionJob {
      name: String,
      schedule: String, // Cron expression
      sources: Vec<String>,
      tickers: TickerSelection, // All, Watchlist, or Specific
      enabled: bool,
      retry_policy: RetryPolicy,
  }
  ```

- [ ] Implement job persistence and recovery:
  - [ ] Store job state in database
  - [ ] Resume interrupted jobs on application restart
  - [ ] Prevent duplicate jobs from running simultaneously
  - [ ] Log all job executions with timing and results

### 2.5 Database Performance Optimization

- [ ] Create indexes for common query patterns:
  ```sql
  -- For algorithm queries
  CREATE INDEX idx_prices_ticker_time ON stock_prices(ticker_id, timestamp DESC);
  CREATE INDEX idx_indicators_ticker_name_time ON technical_indicators(ticker_id, indicator_name, timestamp DESC);
  
  -- For news queries
  CREATE INDEX idx_article_tickers_ticker ON article_tickers(ticker_id, article_id);
  
  -- For lookups
  CREATE INDEX idx_tickers_symbol ON tickers(symbol);
  ```

- [ ] Configure TimescaleDB compression and retention policies:
  ```sql
  -- Compress data older than 7 days
  ALTER TABLE stock_prices SET (
      timescaledb.compress,
      timescaledb.compress_segmentby = 'ticker_id',
      timescaledb.compress_orderby = 'timestamp DESC'
  );
  
  SELECT add_compression_policy('stock_prices', INTERVAL '7 days');
  
  -- Drop raw intraday data older than 2 years (keep compressed)
  SELECT add_retention_policy('stock_prices', INTERVAL '2 years');
  ```

- [ ] Implement connection pool monitoring and tuning:
  - [ ] Monitor pool utilization metrics
  - [ ] Adjust pool size based on workload
  - [ ] Configure statement timeout and idle connection timeout

---

## 3. Algorithm Execution Architecture

### 3.1 Algorithm Framework

**Design Philosophy**: Each algorithm is an independent, composable unit that consumes data and produces trading signals

- [ ] Define core `Algorithm` trait:
  ```rust
  #[async_trait]
  pub trait Algorithm: Send + Sync {
      // Unique identifier for the algorithm
      fn name(&self) -> &str;
      
      // Algorithm version (for tracking performance across versions)
      fn version(&self) -> &str;
      
      // Required data dependencies (prices, indicators, news, etc.)
      fn dependencies(&self) -> Vec<DataDependency>;
      
      // Execute algorithm on data and produce signals
      async fn execute(
          &self,
          context: &AlgorithmContext,
          data: &MarketData,
      ) -> Result<Vec<TradingSignal>>;
      
      // Algorithm-specific parameters (can be tuned)
      fn parameters(&self) -> HashMap<String, ParameterValue>;
      
      // Resource requirements (estimated CPU, memory)
      fn resource_requirements(&self) -> ResourceRequirements;
  }
  ```

- [ ] Create `TradingSignal` data structure:
  ```rust
  pub struct TradingSignal {
      pub ticker: String,
      pub signal_type: SignalType, // Buy, Sell, Hold
      pub confidence: f64, // 0.0 to 1.0
      pub target_position_size: f64, // Percentage of portfolio
      pub time_horizon: TimeHorizon, // Intraday, ShortTerm (days), MediumTerm (weeks), LongTerm (months)
      pub price_target: Option<f64>,
      pub stop_loss: Option<f64>,
      pub reasoning: String,
      pub metadata: HashMap<String, Value>,
      pub generated_at: DateTime<Utc>,
  }
  ```

### 3.2 Parallel Algorithm Execution

**Execution Strategy**: Use Rayon for CPU-intensive computations, Tokio for I/O and coordination

- [ ] Create `AlgorithmExecutor` service with dual thread pools:
  ```rust
  pub struct AlgorithmExecutor {
      // Tokio runtime for I/O (database reads, coordination)
      tokio_runtime: Runtime,
      
      // Rayon thread pool for CPU-intensive algorithm computation
      rayon_pool: ThreadPool,
      
      // Algorithm registry
      algorithms: Vec<Box<dyn Algorithm>>,
      
      // Configuration
      config: ExecutorConfig,
  }
  
  impl AlgorithmExecutor {
      pub fn new(num_cpu_threads: usize) -> Self {
          // Tokio with minimal worker threads (2-4) for I/O only
          let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
              .worker_threads(2)
              .thread_name("tokio-io")
              .build()
              .unwrap();
          
          // Rayon pool using all CPU cores for computation
          let rayon_pool = rayon::ThreadPoolBuilder::new()
              .num_threads(num_cpu_threads)
              .thread_name(|i| format!("rayon-compute-{}", i))
              .build()
              .unwrap();
          
          Self {
              tokio_runtime,
              rayon_pool,
              algorithms: Vec::new(),
              config: ExecutorConfig::default(),
          }
      }
  }
  ```

- [ ] Implement parallel execution workflow:
  ```rust
  pub async fn execute_all_algorithms(
      &self,
      tickers: Vec<String>,
  ) -> Result<Vec<TradingSignal>> {
      // Step 1: Load all required data from database (I/O - Tokio)
      let market_data = self.load_market_data(&tickers).await?;
      
      // Step 2: Execute algorithms in parallel (CPU - Rayon)
      let (tx, rx) = tokio::sync::oneshot::channel();
      let market_data_clone = market_data.clone();
      let algorithms = self.algorithms.clone();
      
      self.rayon_pool.spawn(move || {
          // Run all algorithms in parallel using Rayon
          let signals: Vec<_> = algorithms
              .par_iter()
              .flat_map(|algo| {
                  algo.execute_sync(&market_data_clone)
                      .unwrap_or_default()
              })
              .collect();
          
          tx.send(signals).unwrap();
      });
      
      // Step 3: Collect results (Tokio)
      let all_signals = rx.await?;
      
      // Step 4: Store signals in database for analysis
      self.store_signals(&all_signals).await?;
      
      Ok(all_signals)
  }
  ```

- [ ] Create work-stealing DAG scheduler for dependent algorithms:
  - [ ] Build dependency graph between algorithms
  - [ ] Use topological sort to determine execution order
  - [ ] Execute independent algorithms in parallel
  - [ ] Pass results between dependent algorithms efficiently

### 3.3 Example Algorithm Implementations

- [ ] **Moving Average Crossover Algorithm**:
  - Parameters: Short window (e.g., 50 days), long window (e.g., 200 days)
  - Signal: Buy when short MA crosses above long MA, sell on opposite
  - Confidence: Based on momentum and volume confirmation

- [ ] **RSI Mean Reversion Algorithm**:
  - Parameters: RSI period (14), oversold threshold (30), overbought threshold (70)
  - Signal: Buy when RSI < 30, sell when RSI > 70
  - Confidence: Based on historical RSI reversal success rate

- [ ] **MACD Momentum Algorithm**:
  - Parameters: Fast EMA (12), slow EMA (26), signal line (9)
  - Signal: Buy on MACD line crossing above signal line with positive histogram
  - Confidence: Based on histogram strength and trend

- [ ] **Bollinger Bands Breakout Algorithm**:
  - Parameters: Period (20), standard deviations (2)
  - Signal: Buy on price breaking above upper band, sell below lower band
  - Confidence: Based on volume confirmation and band width

- [ ] **Volume-Weighted Moving Average Algorithm**:
  - Parameters: VWAP period, volume threshold
  - Signal: Buy when price crosses above VWAP with high volume
  - Confidence: Based on volume surge and price momentum

- [ ] **News Sentiment Algorithm**:
  - Dependencies: Recent news articles with sentiment scores
  - Signal: Buy when positive sentiment surge, sell on negative
  - Confidence: Based on sentiment magnitude and source credibility

- [ ] **Fundamental Value Algorithm**:
  - Dependencies: Financial statements, P/E ratio, revenue growth
  - Signal: Buy undervalued stocks with strong fundamentals
  - Time horizon: Long-term (months to years)
  - Confidence: Based on valuation metrics vs. sector averages

- [ ] **Options Flow Algorithm**:
  - Dependencies: Unusual options activity, open interest changes
  - Signal: Follow large institutional options positions
  - Confidence: Based on contract size and timing

### 3.4 Signal Aggregation and Weighting

**Goal**: Combine signals from multiple algorithms with dynamic weighting based on recent performance

- [ ] Create `SignalAggregator` service:
  ```rust
  pub struct SignalAggregator {
      db_pool: Pool,
      weighting_strategy: Box<dyn WeightingStrategy>,
  }
  
  pub trait WeightingStrategy: Send + Sync {
      // Calculate weight for each algorithm based on historical performance
      async fn calculate_weights(
          &self,
          performance_history: &PerformanceHistory,
      ) -> HashMap<String, f64>;
  }
  ```

- [ ] Implement dynamic weighting strategies:
  
  **1. Recent Performance Weighting**:
  - [ ] Track last 30 days of simulated trades per algorithm
  - [ ] Calculate Sharpe ratio, win rate, and average return
  - [ ] Assign higher weight to recently successful algorithms
  - [ ] Exponentially decay weight of older performance
  
  **2. Volatility-Adjusted Weighting**:
  - [ ] Reduce weight of algorithms with high prediction variance
  - [ ] Increase weight of consistent performers
  
  **3. Ensemble Learning Approach**:
  - [ ] Use stacking: meta-algorithm learns optimal weights
  - [ ] Train on historical backtests
  - [ ] Update weights weekly based on rolling window

- [ ] Create aggregated signal scoring:
  ```rust
  pub struct AggregatedSignal {
      pub ticker: String,
      pub composite_score: f64, // Weighted average of all signals
      pub signal_count: usize,
      pub algorithms: Vec<AlgorithmSignal>,
      pub time_horizon: TimeHorizon,
      pub risk_score: f64,
  }
  
  pub fn aggregate_signals(
      signals: Vec<TradingSignal>,
      weights: &HashMap<String, f64>,
  ) -> Vec<AggregatedSignal> {
      // Group signals by ticker
      let grouped = signals.into_iter()
          .into_group_map_by(|s| s.ticker.clone());
      
      grouped.into_iter()
          .map(|(ticker, ticker_signals)| {
              // Calculate weighted composite score
              let composite_score = ticker_signals.iter()
                  .map(|s| {
                      let algo_weight = weights.get(&s.algorithm_name)
                          .unwrap_or(&1.0);
                      s.confidence * algo_weight
                  })
                  .sum::<f64>() / ticker_signals.len() as f64;
              
              AggregatedSignal {
                  ticker,
                  composite_score,
                  signal_count: ticker_signals.len(),
                  algorithms: ticker_signals,
                  // ... other fields
              }
          })
          .collect()
  }
  ```

### 3.5 Performance Feedback Loop

**Strategy**: Continuously evaluate algorithm performance and adjust weights automatically

- [ ] Implement simulated trade evaluation:
  - [ ] For each signal generated, create a "simulated trade" record
  - [ ] Track hypothetical entry price, exit price, and P&L
  - [ ] Calculate metrics: return %, max drawdown, holding period
  - [ ] Compare to buy-and-hold benchmark

- [ ] Create performance tracking database tables:
  ```sql
  CREATE TABLE algorithm_signals (
      signal_id SERIAL PRIMARY KEY,
      algorithm_name VARCHAR(100) NOT NULL,
      algorithm_version VARCHAR(20) NOT NULL,
      ticker VARCHAR(10) NOT NULL,
      signal_type VARCHAR(10) NOT NULL,
      confidence DECIMAL(5,4),
      price_at_signal DECIMAL(12,4),
      generated_at TIMESTAMPTZ NOT NULL,
      metadata JSONB
  );
  
  CREATE TABLE simulated_trades (
      trade_id SERIAL PRIMARY KEY,
      signal_id INTEGER REFERENCES algorithm_signals(signal_id),
      entry_price DECIMAL(12,4),
      exit_price DECIMAL(12,4),
      entry_time TIMESTAMPTZ NOT NULL,
      exit_time TIMESTAMPTZ,
      position_size DECIMAL(10,4),
      return_pct DECIMAL(10,4),
      return_amount DECIMAL(12,4),
      holding_period INTERVAL,
      outcome VARCHAR(20), -- 'win', 'loss', 'neutral'
      metadata JSONB
  );
  
  CREATE TABLE algorithm_performance (
      performance_id SERIAL PRIMARY KEY,
      algorithm_name VARCHAR(100) NOT NULL,
      algorithm_version VARCHAR(20) NOT NULL,
      evaluation_period_start TIMESTAMPTZ NOT NULL,
      evaluation_period_end TIMESTAMPTZ NOT NULL,
      total_signals INTEGER,
      win_rate DECIMAL(5,4),
      avg_return DECIMAL(10,4),
      sharpe_ratio DECIMAL(8,4),
      max_drawdown DECIMAL(10,4),
      current_weight DECIMAL(5,4),
      created_at TIMESTAMPTZ DEFAULT NOW()
  );
  ```

- [ ] Build backtesting engine:
  - [ ] Replay historical signals against actual price movements
  - [ ] Evaluate different exit strategies (time-based, target-based, stop-loss)
  - [ ] Test parameter variations to find optimal settings
  - [ ] Run walk-forward analysis to prevent overfitting

- [ ] Implement automated weight adjustment:
  ```rust
  pub async fn update_algorithm_weights(
      &self,
      lookback_days: i64,
  ) -> Result<HashMap<String, f64>> {
      // Calculate performance for each algorithm
      let performance = self.calculate_recent_performance(lookback_days).await?;
      
      // Apply weighting strategy
      let new_weights = self.weighting_strategy
          .calculate_weights(&performance)
          .await?;
      
      // Store updated weights
      self.store_weights(&new_weights).await?;
      
      // Notify executor of weight changes
      self.notify_weight_update(&new_weights).await?;
      
      Ok(new_weights)
  }
  ```

- [ ] Create continuous evaluation job:
  - [ ] Run daily after market close
  - [ ] Evaluate all signals from previous day(s)
  - [ ] Update performance metrics
  - [ ] Adjust algorithm weights weekly
  - [ ] Flag underperforming algorithms for review

### 3.6 Algorithm Resource Management

- [ ] Implement resource monitoring:
  ```rust
  pub struct AlgorithmMetrics {
      pub execution_time: Duration,
      pub cpu_time: Duration,
      pub memory_used: usize,
      pub database_queries: usize,
  }
  ```

- [ ] Create timeout and circuit breaker mechanisms:
  - [ ] Set per-algorithm execution timeout (e.g., 30 seconds)
  - [ ] Disable algorithms that consistently fail or timeout
  - [ ] Alert on algorithm errors exceeding threshold
  - [ ] Automatically re-enable after cooldown period

---

## 4. Trade Execution Integration

### 4.1 Alpaca Markets Integration

**Alpaca Features**:
- Commission-free trading for stocks, ETFs, and options
- Real-time and historical market data API
- Paper trading environment for testing
- REST API and WebSocket streaming
- Fractional shares support
- Margin trading (up to 4x intraday, 2x overnight)

- [ ] Create `AlpacaClient` module:
  ```rust
  pub struct AlpacaClient {
      api_key: String,
      api_secret: String,
      base_url: String, // Paper or live trading URL
      client: reqwest::Client,
  }
  
  impl AlpacaClient {
      pub fn new(config: AlpacaConfig) -> Self {
          let base_url = if config.paper_trading {
              "https://paper-api.alpaca.markets"
          } else {
              "https://api.alpaca.markets"
          };
          // Implementation
      }
  }
  ```

- [ ] Implement core Alpaca API methods:
  - [ ] `get_account()` - Fetch account details and buying power
  - [ ] `get_positions()` - Get all current positions
  - [ ] `get_orders()` - Fetch order history and status
  - [ ] `create_order()` - Place market/limit/stop orders
  - [ ] `cancel_order()` - Cancel pending orders
  - [ ] `get_asset()` - Check if asset is tradable
  - [ ] `get_bars()` - Fetch historical price bars
  - [ ] `get_latest_trade()` - Get most recent trade data

- [ ] Implement WebSocket streaming for real-time data:
  ```rust
  pub async fn stream_market_data(
      &self,
      tickers: Vec<String>,
      handler: impl FnMut(MarketDataUpdate) + Send + 'static,
  ) -> Result<()> {
      // Connect to Alpaca WebSocket
      // Subscribe to trades, quotes, bars
      // Handle reconnection logic
  }
  ```

### 4.2 Order Management System

- [ ] Create `OrderManager` service:
  ```rust
  pub struct OrderManager {
      alpaca_client: Arc<AlpacaClient>,
      db_pool: Pool,
      risk_manager: Arc<RiskManager>,
      order_queue: Arc<Mutex<VecDeque<PendingOrder>>>,
  }
  ```

- [ ] Implement order types:
  ```rust
  pub enum OrderType {
      Market,
      Limit { limit_price: f64 },
      Stop { stop_price: f64 },
      StopLimit { stop_price: f64, limit_price: f64 },
      TrailingStop { trail_percent: f64 },
  }
  
  pub struct OrderRequest {
      pub ticker: String,
      pub side: OrderSide, // Buy or Sell
      pub quantity: f64, // Can be fractional
      pub order_type: OrderType,
      pub time_in_force: TimeInForce, // Day, GTC, IOC, FOK
      pub extended_hours: bool,
  }
  ```

- [ ] Build order execution workflow:
  1. Receive aggregated trading signal
  2. Pass through risk management validation
  3. Calculate position size based on portfolio allocation rules
  4. Check account buying power
  5. Create order request
  6. Submit to Alpaca API
  7. Monitor order status (filled, partially filled, canceled)
  8. Update internal position tracking
  9. Log all order activity to database

- [ ] Create order tracking database tables:
  ```sql
  CREATE TABLE orders (
      order_id SERIAL PRIMARY KEY,
      alpaca_order_id VARCHAR(50) UNIQUE,
      ticker VARCHAR(10) NOT NULL,
      side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
      quantity DECIMAL(12,4) NOT NULL,
      order_type VARCHAR(20) NOT NULL,
      limit_price DECIMAL(12,4),
      stop_price DECIMAL(12,4),
      status VARCHAR(20) NOT NULL, -- 'pending', 'filled', 'canceled', 'rejected'
      filled_quantity DECIMAL(12,4),
      filled_avg_price DECIMAL(12,4),
      submitted_at TIMESTAMPTZ NOT NULL,
      filled_at TIMESTAMPTZ,
      signal_id INTEGER REFERENCES algorithm_signals(signal_id),
      metadata JSONB
  );
  
  CREATE TABLE positions (
      position_id SERIAL PRIMARY KEY,
      ticker VARCHAR(10) NOT NULL,
      quantity DECIMAL(12,4) NOT NULL,
      avg_entry_price DECIMAL(12,4) NOT NULL,
      current_price DECIMAL(12,4),
      unrealized_pl DECIMAL(12,4),
      unrealized_pl_pct DECIMAL(10,4),
      opened_at TIMESTAMPTZ NOT NULL,
      last_updated TIMESTAMPTZ DEFAULT NOW(),
      metadata JSONB
  );
  ```

### 4.3 Risk Management

**Goal**: Prevent excessive losses and ensure disciplined trading

- [ ] Create `RiskManager` service:
  ```rust
  pub struct RiskManager {
      db_pool: Pool,
      config: RiskConfig,
  }
  
  pub struct RiskConfig {
      pub max_position_size_pct: f64, // Max % of portfolio per position
      pub max_daily_loss_pct: f64, // Stop trading if daily loss exceeds
      pub max_sector_exposure_pct: f64, // Max % in any one sector
      pub max_correlation: f64, // Avoid highly correlated positions
      pub min_liquidity_volume: i64, // Minimum daily volume
  }
  ```

- [ ] Implement risk checks:
  
  **1. Position Size Limits**:
  - [ ] Enforce maximum position size per ticker
  - [ ] Scale position size based on conviction score
  - [ ] Implement Kelly Criterion for optimal sizing
  
  **2. Portfolio-Level Risk**:
  - [ ] Calculate portfolio beta and volatility
  - [ ] Monitor sector concentration
  - [ ] Track correlation between positions
  - [ ] Enforce maximum leverage ratio
  
  **3. Daily Loss Limits**:
  - [ ] Track daily P&L in real-time
  - [ ] Halt trading if daily loss exceeds threshold
  - [ ] Send alert notifications
  - [ ] Require manual override to resume
  
  **4. Liquidity Checks**:
  - [ ] Verify average daily volume meets minimum
  - [ ] Check bid-ask spread
  - [ ] Avoid illiquid stocks that may be hard to exit

- [ ] Create stop-loss and take-profit automation:
  - [ ] Automatically place stop-loss orders on position entry
  - [ ] Implement trailing stops that adjust with price movements
  - [ ] Set profit targets and exit rules
  - [ ] Monitor and adjust stops based on volatility

### 4.4 Paper Trading Mode

- [ ] Implement paper trading toggle:
  - [ ] Use Alpaca paper trading API endpoint
  - [ ] Separate paper trading account credentials
  - [ ] Log all paper trades to separate database tables
  - [ ] Compare paper trading performance to live signals

- [ ] Create paper trading analysis:
  - [ ] Track slippage and execution differences
  - [ ] Measure strategy performance without risk
  - [ ] Test new algorithms in paper mode first
  - [ ] Validate risk management rules

---

## 5. Backend Architecture

### 5.1 Service Architecture

**Design**: Modular microkernel architecture with clear separation of concerns

```
┌─────────────────────────────────────────────────────┐
│              API Layer (REST/WebSocket)             │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Data      │  │  Algorithm   │  │   Order    │  │
│  │  Ingestion  │  │  Execution   │  │ Management │  │
│  │   Service   │  │   Service    │  │  Service   │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Signal    │  │     Risk     │  │  Portfolio │  │
│  │ Aggregation │  │  Management  │  │ Management │  │
│  │   Service   │  │   Service    │  │  Service   │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
├─────────────────────────────────────────────────────┤
│         Database Layer (PostgreSQL/TimescaleDB)     │
└─────────────────────────────────────────────────────┘
```

- [ ] Create main application structure:
  ```rust
  // src/main.rs
  #[tokio::main]
  async fn main() -> Result<()> {
      // Initialize logging
      tracing_subscriber::fmt::init();
      
      // Load configuration
      let config = Config::from_env()?;
      
      // Initialize database pool
      let db_pool = create_db_pool(&config.database).await?;
      
      // Run database migrations
      run_migrations(&db_pool).await?;
      
      // Initialize services
      let services = Services::new(config, db_pool).await?;
      
      // Start background jobs
      start_background_jobs(&services).await?;
      
      // Start API server
      start_api_server(services).await?;
      
      Ok(())
  }
  ```

### 5.2 Configuration Management

- [ ] Use environment variables and configuration files:
  ```rust
  // config/config.toml
  [database]
  host = "localhost"
  port = 5432
  database = "trading_system"
  user = "trader"
  max_connections = 50
  
  [alpaca]
  paper_trading = true
  api_key_env = "ALPACA_API_KEY"
  api_secret_env = "ALPACA_API_SECRET"
  
  [data_sources]
  [data_sources.alpha_vantage]
  enabled = true
  api_key_env = "ALPHA_VANTAGE_API_KEY"
  rate_limit = 75  # requests per minute
  
  [algorithm_executor]
  cpu_threads = 8
  enable_all_algorithms = true
  
  [risk_management]
  max_position_size_pct = 10.0
  max_daily_loss_pct = 5.0
  ```

- [ ] Implement configuration loading with `config` crate:
  - [ ] Support multiple configuration sources (file, env vars, CLI args)
  - [ ] Validate configuration at startup
  - [ ] Support hot-reloading of certain config values
  - [ ] Use strongly-typed configuration structs

### 5.3 REST API Endpoints

- [ ] Use `axum` web framework for high-performance HTTP server:
  ```rust
  pub async fn create_api_router(services: Arc<Services>) -> Router {
      Router::new()
          // Market data endpoints
          .route("/api/tickers", get(list_tickers))
          .route("/api/tickers/:symbol/prices", get(get_price_history))
          .route("/api/tickers/:symbol/indicators", get(get_indicators))
          .route("/api/tickers/:symbol/news", get(get_news))
          
          // Algorithm endpoints
          .route("/api/algorithms", get(list_algorithms))
          .route("/api/algorithms/:name/performance", get(get_algorithm_performance))
          .route("/api/signals", get(get_recent_signals))
          .route("/api/signals/aggregated", get(get_aggregated_signals))
          
          // Trading endpoints
          .route("/api/account", get(get_account_info))
          .route("/api/positions", get(get_positions))
          .route("/api/orders", get(get_orders).post(create_order))
          .route("/api/orders/:id", get(get_order).delete(cancel_order))
          
          // Portfolio endpoints
          .route("/api/portfolio/summary", get(get_portfolio_summary))
          .route("/api/portfolio/performance", get(get_portfolio_performance))
          .route("/api/portfolio/allocation", get(get_allocation))
          
          // Administration
          .route("/api/data-ingestion/status", get(get_ingestion_status))
          .route("/api/data-ingestion/trigger", post(trigger_ingestion))
          .route("/api/health", get(health_check))
          
          .layer(Extension(services))
          .layer(CorsLayer::permissive())
  }
  ```

- [ ] Implement API handlers with proper error handling:
  ```rust
  pub async fn get_aggregated_signals(
      Extension(services): Extension<Arc<Services>>,
      Query(params): Query<SignalQueryParams>,
  ) -> Result<Json<Vec<AggregatedSignal>>, ApiError> {
      let signals = services
          .signal_aggregator
          .get_aggregated_signals(params)
          .await?;
      
      Ok(Json(signals))
  }
  ```

### 5.4 WebSocket Real-Time Updates

- [ ] Implement WebSocket endpoints for live updates:
  - [ ] `/ws/prices` - Stream real-time price updates
  - [ ] `/ws/signals` - Stream new trading signals as generated
  - [ ] `/ws/orders` - Stream order status updates
  - [ ] `/ws/portfolio` - Stream portfolio value changes

- [ ] Use `axum-tungstenite` for WebSocket support:
  ```rust
  pub async fn websocket_handler(
      ws: WebSocketUpgrade,
      Extension(services): Extension<Arc<Services>>,
  ) -> Response {
      ws.on_upgrade(|socket| handle_websocket(socket, services))
  }
  
  async fn handle_websocket(
      socket: WebSocket,
      services: Arc<Services>,
  ) {
      // Subscribe to relevant data streams
      // Forward updates to client
      // Handle client disconnect
  }
  ```

### 5.5 Logging and Observability

- [ ] Implement structured logging with `tracing`:
  ```rust
  use tracing::{info, warn, error, instrument};
  
  #[instrument(skip(db_pool))]
  pub async fn ingest_data(
      db_pool: &Pool,
      source: &str,
      ticker: &str,
  ) -> Result<IngestionResult> {
      info!("Starting data ingestion");
      
      match fetch_and_store_data(db_pool, source, ticker).await {
          Ok(result) => {
              info!(
                  records_inserted = result.inserted,
                  duration_ms = result.duration.as_millis(),
                  "Data ingestion completed"
              );
              Ok(result)
          }
          Err(e) => {
              error!(error = %e, "Data ingestion failed");
              Err(e)
          }
      }
  }
  ```

- [ ] Add metrics collection:
  - [ ] Use `metrics` crate to expose Prometheus metrics
  - [ ] Track: API request latency, database query time, algorithm execution time
  - [ ] Monitor: Active connections, queue lengths, memory usage
  - [ ] Create custom metrics for trading-specific KPIs

- [ ] Implement error tracking:
  - [ ] Categorize errors (transient, fatal, user error)
  - [ ] Log stack traces for debugging
  - [ ] Rate-limit error notifications
  - [ ] Create error budget tracking

### 5.6 Testing Strategy

- [ ] Write unit tests for core business logic:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      
      #[test]
      fn test_signal_aggregation() {
          let signals = vec![
              TradingSignal { /* ... */ },
              TradingSignal { /* ... */ },
          ];
          let weights = hashmap!{
              "algo1" => 0.6,
              "algo2" => 0.4,
          };
          
          let aggregated = aggregate_signals(signals, &weights);
          
          assert_eq!(aggregated.len(), 1);
          assert!(aggregated[0].composite_score > 0.5);
      }
  }
  ```

- [ ] Create integration tests:
  - [ ] Test database operations with test database
  - [ ] Mock external API calls
  - [ ] Test full data ingestion pipeline
  - [ ] Test algorithm execution end-to-end

- [ ] Implement property-based tests with `proptest`:
  - [ ] Test position sizing logic with various inputs
  - [ ] Verify risk management rules hold under all conditions
  - [ ] Test signal aggregation with random combinations

---

## 6. Frontend Requirements

### 6.1 Frontend Technology Stack

**Recommendation**: React with Next.js for server-side rendering and optimal performance

- [ ] Initialize Next.js project with TypeScript
- [ ] Use Tailwind CSS for styling
- [ ] Use `recharts` or `visx` for data visualization
- [ ] Use `swr` or `react-query` for data fetching
- [ ] Use `zustand` or `jotai` for state management

### 6.2 Core Frontend Pages

**Dashboard (Home Page)**:
- [ ] Portfolio summary card (total value, daily P&L, % change)
- [ ] Current positions table with real-time P&L
- [ ] Recent signals list (top opportunities)
- [ ] Algorithm performance summary (top/bottom performers)
- [ ] Market overview (major indices, economic indicators)
- [ ] Recent news feed with sentiment

**Algorithms Page**:
- [ ] List all algorithms with enable/disable toggle
- [ ] Performance metrics table:
  - Win rate, Sharpe ratio, average return
  - Total signals generated, current weight
  - Last execution time, status
- [ ] Individual algorithm detail view:
  - Historical performance chart
  - Parameter configuration
  - Recent signals generated
  - Execution logs
- [ ] Algorithm comparison tool (side-by-side metrics)

**Signals Page**:
- [ ] Table of all recent signals (last 7 days)
  - Ticker, signal type, confidence, algorithm count
  - Price at signal, current price, % change
  - Time horizon, generated timestamp
- [ ] Filter by: Date range, ticker, signal type, time horizon
- [ ] Sort by: Confidence, composite score, signal count
- [ ] Signal detail modal:
  - Contributing algorithms and their individual scores
  - Reasoning text
  - Related news articles
  - Chart with technical indicators

**Trading Page**:
- [ ] Pending orders table with cancel action
- [ ] Order history with filters
- [ ] Quick order entry form (for manual overrides)
- [ ] Risk management status indicators
- [ ] Account information (buying power, margin used)

**Portfolio Page**:
- [ ] Position details table (quantity, cost basis, current value, P&L)
- [ ] Sector allocation pie chart
- [ ] Portfolio performance line chart (vs. benchmark)
- [ ] Correlation heatmap of positions
- [ ] Historical trades log

**Data Management Page**:
- [ ] Data source status indicators (last updated, health)
- [ ] Manual data ingestion triggers
- [ ] Ingestion job logs and history
- [ ] Database statistics (table sizes, row counts)

**Settings Page**:
- [ ] Algorithm configuration (parameters, enable/disable)
- [ ] Risk management settings
- [ ] API credentials management
- [ ] Paper trading toggle
- [ ] Notification preferences

### 6.3 Data Visualization Components

- [ ] Create reusable chart components:
  - [ ] `PriceChart` - Candlestick chart with volume
  - [ ] `PerformanceChart` - Line chart for portfolio/algorithm performance
  - [ ] `SignalTimeline` - Timeline of signals overlaid on price
  - [ ] `AllocationPieChart` - Portfolio allocation by sector/ticker
  - [ ] `HeatmapChart` - Correlation or performance heatmap

- [ ] Implement interactive features:
  - [ ] Zoom and pan on charts
  - [ ] Hover tooltips with detailed information
  - [ ] Toggle indicators on/off
  - [ ] Responsive design for mobile viewing

### 6.4 Real-Time Updates

- [ ] Connect to WebSocket endpoints:
  ```typescript
  import { useEffect, useState } from 'react';
  
  export function useRealtimePrices(tickers: string[]) {
    const [prices, setPrices] = useState<Record<string, number>>({});
    
    useEffect(() => {
      const ws = new WebSocket(`ws://localhost:8080/ws/prices`);
      
      ws.onopen = () => {
        ws.send(JSON.stringify({ subscribe: tickers }));
      };
      
      ws.onmessage = (event) => {
        const update = JSON.parse(event.data);
        setPrices(prev => ({ ...prev, [update.ticker]: update.price }));
      };
      
      return () => ws.close();
    }, [tickers]);
    
    return prices;
  }
  ```

- [ ] Implement automatic data refresh:
  - [ ] Poll for updates every 30 seconds for non-critical data
  - [ ] Use WebSocket for real-time price updates
  - [ ] Show loading states during refresh
  - [ ] Display "last updated" timestamps

### 6.5 API Client

- [ ] Create typed API client:
  ```typescript
  // lib/api-client.ts
  export class TradingApiClient {
    private baseUrl: string;
    
    constructor(baseUrl: string) {
      this.baseUrl = baseUrl;
    }
    
    async getAggregatedSignals(params?: SignalQueryParams): Promise<AggregatedSignal[]> {
      const response = await fetch(`${this.baseUrl}/api/signals/aggregated?${new URLSearchParams(params)}`);
      if (!response.ok) throw new Error('Failed to fetch signals');
      return response.json();
    }
    
    async getAlgorithmPerformance(algorithmName: string): Promise<AlgorithmPerformance> {
      const response = await fetch(`${this.baseUrl}/api/algorithms/${algorithmName}/performance`);
      if (!response.ok) throw new Error('Failed to fetch performance');
      return response.json();
    }
    
    // ... other methods
  }
  ```

---

## 7. Infrastructure and DevOps

### 7.1 Local Development Environment

- [ ] Create `docker-compose.yml` for local development:
  ```yaml
  version: '3.8'
  
  services:
    postgres:
      image: timescale/timescaledb:latest-pg16
      environment:
        POSTGRES_DB: trading_system
        POSTGRES_USER: trader
        POSTGRES_PASSWORD: dev_password
      ports:
        - "5432:5432"
      volumes:
        - postgres_data:/var/lib/postgresql/data
        - ./migrations:/docker-entrypoint-initdb.d
    
    backend:
      build:
        context: .
        dockerfile: Dockerfile.dev
      environment:
        DATABASE_URL: postgres://trader:dev_password@postgres:5432/trading_system
        RUST_LOG: info
      ports:
        - "8080:8080"
      volumes:
        - ./:/app
        - cargo_cache:/usr/local/cargo
      depends_on:
        - postgres
    
    frontend:
      build:
        context: ./frontend
        dockerfile: Dockerfile.dev
      environment:
        NEXT_PUBLIC_API_URL: http://localhost:8080
      ports:
        - "3000:3000"
      volumes:
        - ./frontend:/app
        - /app/node_modules
      depends_on:
        - backend
  
  volumes:
    postgres_data:
    cargo_cache:
  ```

- [ ] Create development scripts:
  - [ ] `scripts/dev-setup.sh` - Initialize development environment
  - [ ] `scripts/run-migrations.sh` - Run database migrations
  - [ ] `scripts/seed-test-data.sh` - Populate test data
  - [ ] `scripts/run-tests.sh` - Run all tests

### 7.2 CI/CD Pipeline (GitHub Actions)

- [ ] Create `.github/workflows/ci.yml`:
  ```yaml
  name: CI
  
  on:
    push:
      branches: [main, develop]
    pull_request:
      branches: [main]
  
  env:
    CARGO_TERM_COLOR: always
    RUSTFLAGS: "-D warnings"
  
  jobs:
    check:
      name: Check
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        
        - name: Install Rust
          uses: dtolnay/rust-toolchain@stable
          with:
            components: rustfmt, clippy
        
        - name: Setup sccache
          uses: mozilla-actions/sccache-action@v0.0.4
        
        - name: Rust Cache
          uses: Swatinem/rust-cache@v2
        
        - name: Check formatting
          run: cargo fmt --all -- --check
        
        - name: Run Clippy
          run: cargo clippy --all-targets --all-features -- -D warnings
    
    test:
      name: Test
      runs-on: ubuntu-latest
      services:
        postgres:
          image: timescale/timescaledb:latest-pg16
          env:
            POSTGRES_PASSWORD: test_password
            POSTGRES_DB: test_db
          ports:
            - 5432:5432
          options: >-
            --health-cmd pg_isready
            --health-interval 10s
            --health-timeout 5s
            --health-retries 5
      
      steps:
        - uses: actions/checkout@v4
        
        - name: Install Rust
          uses: dtolnay/rust-toolchain@stable
        
        - name: Setup sccache
          uses: mozilla-actions/sccache-action@v0.0.4
        
        - name: Rust Cache
          uses: Swatinem/rust-cache@v2
        
        - name: Install cargo-nextest
          uses: taiki-e/install-action@cargo-nextest
        
        - name: Run tests
          run: cargo nextest run --all-features
          env:
            DATABASE_URL: postgres://postgres:test_password@localhost:5432/test_db
    
    build:
      name: Build
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        
        - name: Install Rust
          uses: dtolnay/rust-toolchain@stable
        
        - name: Setup sccache
          uses: mozilla-actions/sccache-action@v0.0.4
        
        - name: Rust Cache
          uses: Swatinem/rust-cache@v2
        
        - name: Build release
          run: cargo build --release
        
        - name: Upload binary
          uses: actions/upload-artifact@v4
          with:
            name: trading-system-binary
            path: target/release/trading-system
  ```

- [ ] Create `.github/workflows/frontend-ci.yml` for frontend testing:
  - Build Next.js application
  - Run ESLint and TypeScript checks
  - Run frontend tests (if any)

### 7.3 Database Migrations

- [ ] Use `sqlx-cli` for migrations:
  ```bash
  # Install sqlx-cli
  cargo install sqlx-cli --no-default-features --features postgres
  
  # Create new migration
  sqlx migrate add create_initial_schema
  
  # Run migrations
  sqlx migrate run
  ```

- [ ] Create migration workflow:
  - [ ] Migrations numbered sequentially (001_initial.sql, 002_add_options.sql)
  - [ ] Each migration is idempotent (can be run multiple times)
  - [ ] Include rollback scripts for critical migrations
  - [ ] Test migrations on copy of production data

### 7.4 Secrets Management

- [ ] Store sensitive credentials securely:
  - [ ] Use `.env` file for local development (gitignored)
  - [ ] Use GitHub Secrets for CI/CD
  - [ ] For production: Use environment variables or secret manager
  - [ ] Never commit API keys or passwords to repository

- [ ] Create `.env.example` template:
  ```
  # Database
  DATABASE_URL=postgres://user:password@localhost:5432/trading_system
  
  # Alpaca
  ALPACA_API_KEY=your_key_here
  ALPACA_API_SECRET=your_secret_here
  ALPACA_PAPER_TRADING=true
  
  # Data Sources
  ALPHA_VANTAGE_API_KEY=your_key_here
  FINNHUB_API_KEY=your_key_here
  # ... other API keys
  ```

### 7.5 Deployment

**Deployment Options**:
- Local server (dedicated machine or home lab)
- Cloud VM (AWS EC2, GCP Compute Engine, DigitalOcean Droplet)
- Containerized (Docker + Docker Compose)

- [ ] Create production `Dockerfile`:
  ```dockerfile
  # Multi-stage build for smaller image
  FROM rust:1.75 as builder
  WORKDIR /app
  COPY Cargo.toml Cargo.lock ./
  COPY src ./src
  RUN cargo build --release
  
  FROM debian:bookworm-slim
  RUN apt-get update && apt-get install -y \
      ca-certificates \
      libpq5 \
      && rm -rf /var/lib/apt/lists/*
  
  COPY --from=builder /app/target/release/trading-system /usr/local/bin/
  EXPOSE 8080
  CMD ["trading-system"]
  ```

- [ ] Create deployment script:
  ```bash
  #!/bin/bash
  # scripts/deploy.sh
  
  set -e
  
  echo "Building Docker image..."
  docker build -t trading-system:latest .
  
  echo "Stopping old container..."
  docker stop trading-system || true
  docker rm trading-system || true
  
  echo "Starting new container..."
  docker run -d \
    --name trading-system \
    --env-file .env.production \
    -p 8080:8080 \
    --restart unless-stopped \
    trading-system:latest
  
  echo "Deployment complete!"
  ```

### 7.6 Monitoring and Alerting

- [ ] Set up monitoring:
  - [ ] Log aggregation: Send logs to file or external service
  - [ ] Metrics: Export Prometheus metrics, visualize with Grafana
  - [ ] Uptime monitoring: Simple HTTP health check endpoint
  - [ ] Database monitoring: Track query performance, connection pool usage

- [ ] Create alerting rules:
  - [ ] Alert on: High error rate, service down, database connection failures
  - [ ] Alert on: Daily loss exceeds threshold, position at risk
  - [ ] Alert on: Data ingestion failures, stale data
  - [ ] Delivery: Email, Telegram, or push notifications

- [ ] Implement health check endpoint:
  ```rust
  pub async fn health_check(
      Extension(services): Extension<Arc<Services>>,
  ) -> Result<Json<HealthStatus>, StatusCode> {
      let db_healthy = services.db_pool.status().available > 0;
      let alpaca_healthy = services.alpaca_client.is_connected().await;
      
      let status = HealthStatus {
          healthy: db_healthy && alpaca_healthy,
          database: db_healthy,
          alpaca: alpaca_healthy,
          timestamp: Utc::now(),
      };
      
      if status.healthy {
          Ok(Json(status))
      } else {
          Err(StatusCode::SERVICE_UNAVAILABLE)
      }
  }
  ```

### 7.7 Backup and Disaster Recovery

- [ ] Implement database backup strategy:
  - [ ] Automated daily backups using `pg_dump`
  - [ ] Retain backups for 30 days
  - [ ] Test restore process quarterly
  - [ ] Store backups in separate location

- [ ] Create backup script:
  ```bash
  #!/bin/bash
  # scripts/backup-database.sh
  
  BACKUP_DIR="/backups"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  BACKUP_FILE="$BACKUP_DIR/trading_db_$TIMESTAMP.sql"
  
  pg_dump -h localhost -U trader trading_system > "$BACKUP_FILE"
  gzip "$BACKUP_FILE"
  
  # Keep only last 30 days of backups
  find "$BACKUP_DIR" -name "trading_db_*.sql.gz" -mtime +30 -delete
  
  echo "Backup completed: $BACKUP_FILE.gz"
  ```

- [ ] Document recovery procedures:
  - Step-by-step guide to restore from backup
  - List of critical configuration files to preserve
  - Emergency contact information

---

## 8. Documentation

- [ ] Create comprehensive README.md:
  - Project overview and architecture
  - Setup instructions
  - Configuration guide
  - Running locally vs. production

- [ ] Write API documentation:
  - Use OpenAPI/Swagger for REST API
  - Document all endpoints with examples
  - Include rate limits and error codes

- [ ] Document algorithms:
  - Each algorithm has its own doc file
  - Explain logic, parameters, and expected behavior
  - Include backtesting results and performance analysis

- [ ] Create runbook for operations:
  - How to add new data sources
  - How to add new algorithms
  - Troubleshooting common issues
  - Monitoring and alerting guide

---

## Summary

This specification provides a complete roadmap for building a professional-grade retail trading system. The architecture prioritizes:

1. **Data aggregation** from multiple free and affordable sources
2. **Parallel processing** using Rust's async (Tokio) for I/O and Rayon for CPU-intensive algorithms
3. **Performance feedback** through continuous backtesting and weight adjustment
4. **Risk management** to prevent losses and ensure disciplined trading
5. **Modular design** allowing easy addition of new algorithms and data sources
6. **Developer ergonomics** with a single developer in mind, avoiding unnecessary complexity

The system can be implemented iteratively, starting with core data ingestion and a few simple algorithms, then expanding over time. Each section above can be converted into individual feature tickets and developed serially.

**Estimated Development Timeline** (for experienced Rust developer):
- Core data ingestion + database setup: 2-3 weeks
- Algorithm framework + first 3 algorithms: 2-3 weeks
- Signal aggregation + backtesting: 1-2 weeks
- Alpaca integration + order management: 1-2 weeks
- Risk management: 1 week
- Backend API + basic frontend: 2-3 weeks
- CI/CD + deployment: 1 week
- Testing + refinement: 2-3 weeks

**Total**: ~3-4 months of focused development

**Next Steps**:
1. Set up development environment (PostgreSQL + Rust toolchain)
2. Initialize Git repository and project structure
3. Start with data ingestion for one source (e.g., Alpha Vantage)
4. Build out database schema
5. Implement first algorithm (e.g., Moving Average Crossover)
6. Continue iterating through feature tickets
