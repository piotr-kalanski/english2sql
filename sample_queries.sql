[
    {
        "description": "total revenue by fiscal month",
        "sql": "select d.fiscal_month, sum(f.sales) as revenue from core.profitability_fact f join core.date d on f.date_fk = d.date_key"
    },
    {
        "description": "active customers by fiscal month",
        "sql": "select d.fiscal_month, count(distinct f.customer_fk) as customer_count from core.profitability_fact f join core.date d on f.date_fk = d.date_key where f.sales > 0"
    },
    {
        "description": "order count by customer",
        "sql": "SELECT customer_id, COUNT(order_id) AS order_count FROM main.orders GROUP BY customer_id"
    }
]