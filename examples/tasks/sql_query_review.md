# Task: Review and Improve SQL Query

**Goal**  
Identify issues and propose a corrected query. Explain the rationale briefly.

**Current Query (suspect)**
```sql
SELECT * FROM users u
JOIN orders o
WHERE u.id = o.user_id
GROUP BY u.id
ORDER BY count(o.id) DESC;
