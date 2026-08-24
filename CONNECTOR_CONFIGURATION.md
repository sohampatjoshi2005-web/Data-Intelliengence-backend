# Connector Configuration Guide

Complete reference for configuring all supported connectors in AutoML Platform.

## Database Connectors

### SQLite
```json
{
  "connector": "sqlite",
  "config": {
    "path": "/absolute/path/to/database.db"
  }
}
```

### PostgreSQL
```json
{
  "connector": "postgresql",
  "config": {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "pass"
  }
}
```
Or with connection string:
```json
{
  "connector": "postgresql",
  "config": {
    "url": "postgresql+psycopg2://user:pass@localhost:5432/mydb"
  }
}
```

### MySQL
```json
{
  "connector": "mysql",
  "config": {
    "host": "localhost",
    "port": 3306,
    "database": "mydb",
    "username": "user",
    "password": "pass"
  }
}
```

### Microsoft SQL Server
Requires ODBC Driver 17 for SQL Server. Install on macOS:
```bash
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew install msodbcsql17
```

Configuration:
```json
{
  "connector": "mssql",
  "config": {
    "host": "server.database.windows.net",
    "port": 1433,
    "database": "mydb",
    "username": "user@server",
    "password": "pass",
    "driver": "mssql+pyodbc"
  }
}
```

### Oracle
Requires Oracle Instant Client. Installation varies by OS.

```json
{
  "connector": "oracle",
  "config": {
    "host": "localhost",
    "port": 1521,
    "database": "XE",
    "username": "admin",
    "password": "pass"
  }
}
```

### Amazon Redshift
```json
{
  "connector": "redshift",
  "config": {
    "host": "cluster-name.123456.us-east-1.redshift.amazonaws.com",
    "port": 5439,
    "database": "devdb",
    "username": "awsuser",
    "password": "pass"
  }
}
```

---

## Data Warehouse Connectors

### Snowflake
```json
{
  "connector": "snowflake",
  "config": {
    "account": "xy12345.us-east-1",
    "user": "user@company.com",
    "password": "SecurePassword123",
    "database": "ANALYTICS",
    "warehouse": "COMPUTE",
    "schema": "PUBLIC"
  }
}
```

### BigQuery
Option 1 - Service Account JSON:
```json
{
  "connector": "bigquery",
  "config": {
    "project_id": "my-project",
    "dataset": "my_dataset",
    "service_account_json": {
      "type": "service_account",
      "project_id": "my-project",
      "private_key_id": "...",
      "private_key": "...",
      "client_email": "sa@my-project.iam.gserviceaccount.com",
      "client_id": "...",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token"
    }
  }
}
```

Option 2 - Default Application Credentials:
```json
{
  "connector": "bigquery",
  "config": {
    "project_id": "my-project",
    "dataset": "my_dataset"
  }
}
```

---

## Cloud Storage Connectors

### AWS S3
```json
{
  "connector": "rest_api",
  "config": {
    "url": "https://bucket-name.s3.amazonaws.com/key.csv",
    "headers": {
      "Authorization": "AWS4-HMAC-SHA256 ..."
    }
  }
}
```

Or configure via boto3 in code.

### Google Cloud Storage
```json
{
  "connector": "rest_api",
  "config": {
    "url": "https://storage.googleapis.com/bucket/object",
    "headers": {
      "Authorization": "Bearer access_token"
    }
  }
}
```

### Azure Blob Storage
```json
{
  "connector": "rest_api",
  "config": {
    "url": "https://storageaccount.blob.core.windows.net/container/blob",
    "headers": {
      "Authorization": "SharedKey storageaccount:signature"
    }
  }
}
```

---

## SaaS Connectors (Coming Soon)

These require OAuth2 setup:

### Salesforce
```json
{
  "connector": "salesforce",
  "config": {
    "instance_url": "https://your-org.salesforce.com",
    "client_id": "...",
    "client_secret": "...",
    "username": "user@company.com",
    "password": "pass",
    "security_token": "token"
  }
}
```

### HubSpot
```json
{
  "connector": "hubspot",
  "config": {
    "api_key": "pat-na1-...",
    "object": "contacts"
  }
}
```

### Google Analytics
```json
{
  "connector": "google_analytics",
  "config": {
    "view_id": "VIEW_ID",
    "service_account_json": {...}
  }
}
```

---

## Integration Connectors

### REST API
Any REST endpoint returning JSON:

```json
{
  "connector": "rest_api",
  "config": {
    "url": "https://api.example.com/data",
    "method": "GET",
    "headers": {
      "Authorization": "Bearer token",
      "Content-Type": "application/json"
    },
    "params": {
      "limit": 1000
    },
    "root_key": "data.items"
  }
}
```

Fields:
- `url` (required): API endpoint
- `method` (default: GET): HTTP method
- `headers`: HTTP headers
- `params`: Query parameters
- `body`: JSON body for POST/PUT
- `root_key`: Path to data array in response (dot notation)

### Streaming Connectors (Planned)

**Kafka**
```json
{
  "connector": "kafka",
  "config": {
    "bootstrap_servers": "localhost:9092",
    "topic": "my-topic",
    "group_id": "consumer-group",
    "security_protocol": "PLAINTEXT"
  }
}
```

**WebSocket**
```json
{
  "connector": "websocket",
  "config": {
    "url": "wss://stream.example.com/feed",
    "auth": "Bearer token"
  }
}
```

---

## Usage Examples

### Example 1: Load from PostgreSQL
```bash
curl -X POST "http://localhost:8000/connectors/load" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": "postgresql",
    "config": {
      "host": "localhost",
      "database": "analytics",
      "username": "user",
      "password": "pass"
    },
    "table": "customers",
    "limit": 1000
  }'
```

### Example 2: Load from BigQuery with SQL
```bash
curl -X POST "http://localhost:8000/connectors/load" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": "bigquery",
    "config": {
      "project_id": "my-project",
      "dataset": "analytics"
    },
    "query": "SELECT * FROM users WHERE created_at > TIMESTAMP(\u00272024-01-01\u0027) LIMIT 10000"
  }'
```

### Example 3: Load from REST API
```bash
curl -X POST "http://localhost:8000/connectors/load" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": "rest_api",
    "config": {
      "url": "https://jsonplaceholder.typicode.com/posts",
      "root_key": ""
    },
    "limit": 100
  }'
```

---

## Error Handling

Common errors and solutions:

### Connection Refused
- Check if database server is running
- Verify host/port are correct
- Check firewall rules

### Authentication Failed
- Verify username/password
- Check for special characters that need escaping
- Ensure user has proper permissions

### Table Not Found
- Check table name spelling (case-sensitive in some databases)
- Ensure schema is specified if required
- Verify user has table access

### Timeout
- Reduce LIMIT to smaller number
- Check network connectivity
- Ensure database is responsive

---

## Performance Tips

1. **Use LIMIT**: Always test with LIMIT first before fetching large datasets
2. **Add WHERE clause**: Filter on database side, not in Python
3. **Use Indexes**: Ensure frequently queried columns are indexed
4. **Pre-filter dates**: For time-series data, filter by date range
5. **Connection pooling**: Database connections are pooled automatically

---

## Supported Operations

Each connector supports:
- **Test**: `/connectors/test` - Verify connection works
- **Load**: `/connectors/load` - Fetch data with optional query
- **Preview**: View sample rows before full load
- **AutoML**: Build ML pipeline directly from connector

