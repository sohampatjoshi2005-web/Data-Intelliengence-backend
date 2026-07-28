from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Connector:
    name: str
    category: str
    status: str


CONNECTORS: list[Connector] = [
    Connector("CSV/TSV/TXT", "files", "ready"),
    Connector("Excel", "files", "ready"),
    Connector("JSON", "files", "ready"),
    Connector("PDF (limited extract)", "files", "planned"),
    Connector("SAS (.sas7bdat)", "files", "ready"),
    Connector("SPSS (.sav)", "files", "ready"),
    Connector("RData (.rdata)", "files", "ready"),
    Connector("SQLite", "database", "ready"),
    Connector("PostgreSQL", "database", "ready"),
    Connector("SQL (SQLAlchemy URL)", "database", "ready"),
    Connector("MySQL", "database", "ready"),
    Connector("Microsoft SQL Server", "database", "ready"),
    Connector("Oracle", "database", "ready"),
    Connector("Amazon RDS", "database", "ready"),
    Connector("IBM DB2", "database", "planned"),
    Connector("SAP HANA", "database", "planned"),
    Connector("Teradata", "database", "planned"),
    Connector("MariaDB", "database", "ready"),
    Connector("Snowflake", "warehouse", "ready"),
    Connector("BigQuery", "warehouse", "ready"),
    Connector("Redshift", "warehouse", "ready"),
    Connector("Databricks", "warehouse", "planned"),
    Connector("Azure Synapse", "warehouse", "planned"),
    Connector("Azure SQL DW", "warehouse", "planned"),
    Connector("Hive", "bigdata", "planned"),
    Connector("HDFS", "bigdata", "planned"),
    Connector("Impala", "bigdata", "planned"),
    Connector("Spark SQL", "bigdata", "ready"),
    Connector("Apache Spark", "bigdata", "ready"),
    Connector("MongoDB", "unstructured", "planned"),
    Connector("DynamoDB", "unstructured", "planned"),
    Connector("Firebase", "unstructured", "planned"),
    Connector("Cassandra", "unstructured", "planned"),
    Connector("Google Sheets", "cloud", "planned"),
    Connector("OneDrive/SharePoint", "cloud", "planned"),
    Connector("Dropbox", "cloud", "planned"),
    Connector("Box", "cloud", "planned"),
    Connector("Salesforce", "saas", "planned"),
    Connector("Google Analytics (GA4)", "saas", "planned"),
    Connector("ServiceNow", "saas", "planned"),
    Connector("SAP BW", "saas", "planned"),
    Connector("Marketo", "saas", "planned"),
    Connector("QuickBooks", "saas", "planned"),
    Connector("HubSpot", "saas", "planned"),
    Connector("Facebook Ads", "saas", "planned"),
    Connector("LinkedIn Ads", "saas", "planned"),
    Connector("OData", "integration", "planned"),
    Connector("Web Data Connector (WDC)", "integration", "planned"),
    Connector("REST API", "integration", "ready"),
]


def list_connectors() -> list[dict]:
    return [c.__dict__ for c in CONNECTORS]
