package db

import (
	"context"
	"database/sql"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"
)

const (
	defaultTimeout   = 30 * time.Second
	maxBatchSize     = 1000
	defaultPageSize  = 50
	maxPageSize      = 500
	schemaVersion    = 7
)

type Dialect int

const (
	PostgreSQL Dialect = iota
	MySQL
	SQLite
)

type QueryType int

const (
	SelectQuery QueryType = iota
	InsertQuery
	UpdateQuery
	DeleteQuery
	UpsertQuery
)

type JoinType string

const (
	InnerJoin JoinType = "INNER JOIN"
	LeftJoin  JoinType = "LEFT JOIN"
	RightJoin JoinType = "RIGHT JOIN"
	FullJoin  JoinType = "FULL OUTER JOIN"
	CrossJoin JoinType = "CROSS JOIN"
)

type OrderDir string

const (
	ASC  OrderDir = "ASC"
	DESC OrderDir = "DESC"
)

type Expr struct {
	sql  string
	args []interface{}
}

func Raw(sql string, args ...interface{}) Expr {
	return Expr{sql: sql, args: args}
}

func Eq(col string, val interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s = ?", col), args: []interface{}{val}}
}

func NEq(col string, val interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s != ?", col), args: []interface{}{val}}
}

func Gt(col string, val interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s > ?", col), args: []interface{}{val}}
}

func Gte(col string, val interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s >= ?", col), args: []interface{}{val}}
}

func Lt(col string, val interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s < ?", col), args: []interface{}{val}}
}

func Lte(col string, val interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s <= ?", col), args: []interface{}{val}}
}

func Like(col, pattern string) Expr {
	return Expr{sql: fmt.Sprintf("%s LIKE ?", col), args: []interface{}{pattern}}
}

func ILike(col, pattern string) Expr {
	return Expr{sql: fmt.Sprintf("%s ILIKE ?", col), args: []interface{}{pattern}}
}

func In(col string, vals ...interface{}) Expr {
	placeholders := make([]string, len(vals))
	for i := range vals {
		placeholders[i] = "?"
	}
	return Expr{
		sql:  fmt.Sprintf("%s IN (%s)", col, strings.Join(placeholders, ", ")),
		args: vals,
	}
}

func NotIn(col string, vals ...interface{}) Expr {
	placeholders := make([]string, len(vals))
	for i := range vals {
		placeholders[i] = "?"
	}
	return Expr{
		sql:  fmt.Sprintf("%s NOT IN (%s)", col, strings.Join(placeholders, ", ")),
		args: vals,
	}
}

func IsNull(col string) Expr {
	return Expr{sql: fmt.Sprintf("%s IS NULL", col)}
}

func IsNotNull(col string) Expr {
	return Expr{sql: fmt.Sprintf("%s IS NOT NULL", col)}
}

func Between(col string, lo, hi interface{}) Expr {
	return Expr{sql: fmt.Sprintf("%s BETWEEN ? AND ?", col), args: []interface{}{lo, hi}}
}

func And(exprs ...Expr) Expr {
	parts := make([]string, len(exprs))
	var args []interface{}
	for i, e := range exprs {
		parts[i] = "(" + e.sql + ")"
		args = append(args, e.args...)
	}
	return Expr{sql: strings.Join(parts, " AND "), args: args}
}

func Or(exprs ...Expr) Expr {
	parts := make([]string, len(exprs))
	var args []interface{}
	for i, e := range exprs {
		parts[i] = "(" + e.sql + ")"
		args = append(args, e.args...)
	}
	return Expr{sql: strings.Join(parts, " OR "), args: args}
}

func Not(expr Expr) Expr {
	return Expr{sql: "NOT (" + expr.sql + ")", args: expr.args}
}

type joinClause struct {
	joinType JoinType
	table    string
	on       Expr
}

type orderClause struct {
	column string
	dir    OrderDir
}

type QueryBuilder struct {
	queryType  QueryType
	table      string
	alias      string
	columns    []string
	joins      []joinClause
	conditions []Expr
	groupBy    []string
	having     []Expr
	orderBy    []orderClause
	limit      int
	offset     int
	lockMode   string
	withCTE    map[string]string
	setValues  map[string]interface{}
	insertRows []map[string]interface{}
	dialect    Dialect
	mu         sync.RWMutex
}

func NewQuery(table string) *QueryBuilder {
	return &QueryBuilder{
		queryType: SelectQuery,
		table:     table,
		withCTE:   make(map[string]string),
		setValues: make(map[string]interface{}),
		dialect:   PostgreSQL,
	}
}

func (q *QueryBuilder) As(alias string) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.alias = alias
	return q
}

func (q *QueryBuilder) Select(cols ...string) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.queryType = SelectQuery
	q.columns = append(q.columns, cols...)
	return q
}

func (q *QueryBuilder) From(table string) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.table = table
	return q
}

func (q *QueryBuilder) Join(table string, on Expr) *QueryBuilder {
	return q.addJoin(InnerJoin, table, on)
}

func (q *QueryBuilder) LeftJoin(table string, on Expr) *QueryBuilder {
	return q.addJoin(LeftJoin, table, on)
}

func (q *QueryBuilder) RightJoin(table string, on Expr) *QueryBuilder {
	return q.addJoin(RightJoin, table, on)
}

func (q *QueryBuilder) FullJoin(table string, on Expr) *QueryBuilder {
	return q.addJoin(FullJoin, table, on)
}

func (q *QueryBuilder) addJoin(jt JoinType, table string, on Expr) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.joins = append(q.joins, joinClause{joinType: jt, table: table, on: on})
	return q
}

func (q *QueryBuilder) Where(exprs ...Expr) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.conditions = append(q.conditions, exprs...)
	return q
}

func (q *QueryBuilder) GroupBy(cols ...string) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.groupBy = append(q.groupBy, cols...)
	return q
}

func (q *QueryBuilder) Having(exprs ...Expr) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.having = append(q.having, exprs...)
	return q
}

func (q *QueryBuilder) OrderBy(col string, dir OrderDir) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.orderBy = append(q.orderBy, orderClause{column: col, dir: dir})
	return q
}

func (q *QueryBuilder) Limit(n int) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	if n > maxPageSize { n = maxPageSize }
	q.limit = n
	return q
}

func (q *QueryBuilder) Offset(n int) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.offset = n
	return q
}

func (q *QueryBuilder) Page(page, size int) *QueryBuilder {
	if size <= 0 { size = defaultPageSize }
	if size > maxPageSize { size = maxPageSize }
	if page <= 0 { page = 1 }
	return q.Limit(size).Offset((page - 1) * size)
}

func (q *QueryBuilder) ForUpdate() *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.lockMode = "FOR UPDATE"
	return q
}

func (q *QueryBuilder) ForShare() *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.lockMode = "FOR SHARE"
	return q
}

func (q *QueryBuilder) With(name, subquery string) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.withCTE[name] = subquery
	return q
}

func (q *QueryBuilder) Set(col string, val interface{}) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.queryType = UpdateQuery
	q.setValues[col] = val
	return q
}

func (q *QueryBuilder) Insert(rows ...map[string]interface{}) *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.queryType = InsertQuery
	q.insertRows = append(q.insertRows, rows...)
	return q
}

func (q *QueryBuilder) Delete() *QueryBuilder {
	q.mu.Lock(); defer q.mu.Unlock()
	q.queryType = DeleteQuery
	return q
}

func (q *QueryBuilder) Build() (string, []interface{}) {
	q.mu.RLock(); defer q.mu.RUnlock()
	var sb strings.Builder
	var args []interface{}

	if len(q.withCTE) > 0 {
		sb.WriteString("WITH ")
		parts := make([]string, 0, len(q.withCTE))
		for name, sub := range q.withCTE {
			parts = append(parts, fmt.Sprintf("%s AS (%s)", name, sub))
		}
		sb.WriteString(strings.Join(parts, ", "))
		sb.WriteString(" ")
	}

	switch q.queryType {
	case SelectQuery:
		cols := "*"
		if len(q.columns) > 0 {
			cols = strings.Join(q.columns, ", ")
		}
		tableExpr := q.table
		if q.alias != "" {
			tableExpr += " AS " + q.alias
		}
		fmt.Fprintf(&sb, "SELECT %s FROM %s", cols, tableExpr)
		for _, j := range q.joins {
			fmt.Fprintf(&sb, " %s %s ON %s", j.joinType, j.table, j.on.sql)
			args = append(args, j.on.args...)
		}
	case InsertQuery:
		if len(q.insertRows) == 0 { break }
		keys := make([]string, 0)
		for k := range q.insertRows[0] {
			keys = append(keys, k)
		}
		placeholders := make([]string, len(keys))
		for i := range keys { placeholders[i] = "?" }
		fmt.Fprintf(&sb, "INSERT INTO %s (%s) VALUES", q.table, strings.Join(keys, ", "))
		rowExprs := make([]string, len(q.insertRows))
		for i, row := range q.insertRows {
			rowExprs[i] = "(" + strings.Join(placeholders, ", ") + ")"
			for _, k := range keys {
				args = append(args, row[k])
			}
		}
		sb.WriteString(" " + strings.Join(rowExprs, ", "))
	case UpdateQuery:
		sb.WriteString("UPDATE " + q.table + " SET ")
		setParts := make([]string, 0, len(q.setValues))
		for col, val := range q.setValues {
			setParts = append(setParts, col+" = ?")
			args = append(args, val)
		}
		sb.WriteString(strings.Join(setParts, ", "))
	case DeleteQuery:
		sb.WriteString("DELETE FROM " + q.table)
	}

	if len(q.conditions) > 0 {
		parts := make([]string, len(q.conditions))
		for i, c := range q.conditions {
			parts[i] = c.sql
			args = append(args, c.args...)
		}
		sb.WriteString(" WHERE " + strings.Join(parts, " AND "))
	}

	if len(q.groupBy) > 0 {
		sb.WriteString(" GROUP BY " + strings.Join(q.groupBy, ", "))
	}

	if len(q.having) > 0 {
		parts := make([]string, len(q.having))
		for i, h := range q.having {
			parts[i] = h.sql
			args = append(args, h.args...)
		}
		sb.WriteString(" HAVING " + strings.Join(parts, " AND "))
	}

	if len(q.orderBy) > 0 {
		parts := make([]string, len(q.orderBy))
		for i, o := range q.orderBy {
			parts[i] = fmt.Sprintf("%s %s", o.column, o.dir)
		}
		sb.WriteString(" ORDER BY " + strings.Join(parts, ", "))
	}

	if q.limit > 0 {
		fmt.Fprintf(&sb, " LIMIT %d", q.limit)
	}
	if q.offset > 0 {
		fmt.Fprintf(&sb, " OFFSET %d", q.offset)
	}
	if q.lockMode != "" {
		sb.WriteString(" " + q.lockMode)
	}

	query := sb.String()
	if q.dialect == PostgreSQL {
		query = rebindPostgres(query)
	}
	return query, args
}

func rebindPostgres(query string) string {
	var sb strings.Builder
	n := 1
	for _, ch := range query {
		if ch == '?' {
			fmt.Fprintf(&sb, "$%d", n)
			n++
		} else {
			sb.WriteRune(ch)
		}
	}
	return sb.String()
}

type Repository[T any] struct {
	db      *sql.DB
	table   string
	dialect Dialect
}

func NewRepository[T any](db *sql.DB, table string) *Repository[T] {
	return &Repository[T]{db: db, table: table, dialect: PostgreSQL}
}

func (r *Repository[T]) FindByID(ctx context.Context, id interface{}) (*T, error) {
	q, args := NewQuery(r.table).
		Where(Eq("id", id)).
		Limit(1).
		Build()
	row := r.db.QueryRowContext(ctx, q, args...)
	var result T
	if err := scanStruct(row, &result); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("FindByID: %w", err)
	}
	return &result, nil
}

func (r *Repository[T]) FindAll(ctx context.Context, opts ...func(*QueryBuilder)) ([]T, error) {
	qb := NewQuery(r.table)
	for _, opt := range opts {
		opt(qb)
	}
	q, args := qb.Build()
	rows, err := r.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("FindAll: %w", err)
	}
	defer rows.Close()
	var results []T
	for rows.Next() {
		var item T
		if err := scanStruct(rows, &item); err != nil {
			return nil, err
		}
		results = append(results, item)
	}
	return results, rows.Err()
}

func (r *Repository[T]) Count(ctx context.Context, exprs ...Expr) (int64, error) {
	qb := NewQuery(r.table).Select("COUNT(*)")
	if len(exprs) > 0 {
		qb.Where(exprs...)
	}
	q, args := qb.Build()
	var count int64
	if err := r.db.QueryRowContext(ctx, q, args...).Scan(&count); err != nil {
		return 0, fmt.Errorf("Count: %w", err)
	}
	return count, nil
}

func (r *Repository[T]) Exists(ctx context.Context, exprs ...Expr) (bool, error) {
	count, err := r.Count(ctx, exprs...)
	return count > 0, err
}

func (r *Repository[T]) BatchInsert(ctx context.Context, items []T) error {
	if len(items) == 0 { return nil }
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil { return err }
	defer func() {
		if err != nil { _ = tx.Rollback() }
	}()
	for i := 0; i < len(items); i += maxBatchSize {
		end := i + maxBatchSize
		if end > len(items) { end = len(items) }
		batch := items[i:end]
		rows := make([]map[string]interface{}, len(batch))
		for j, item := range batch {
			rows[j] = structToMap(item)
		}
		q, args := NewQuery(r.table).Insert(rows...).Build()
		if _, err = tx.ExecContext(ctx, q, args...); err != nil {
			return fmt.Errorf("BatchInsert chunk %d: %w", i/maxBatchSize, err)
		}
	}
	return tx.Commit()
}

func scanStruct(scanner interface{ Scan(...interface{}) error }, dest interface{}) error {
	v := reflect.ValueOf(dest).Elem()
	t := v.Type()
	ptrs := make([]interface{}, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		ptrs[i] = v.Field(i).Addr().Interface()
	}
	return scanner.Scan(ptrs...)
}

func structToMap(v interface{}) map[string]interface{} {
	rv := reflect.ValueOf(v)
	rt := rv.Type()
	m := make(map[string]interface{}, rt.NumField())
	for i := 0; i < rt.NumField(); i++ {
		field := rt.Field(i)
		tag := field.Tag.Get("db")
		if tag == "" || tag == "-" { continue }
		m[tag] = rv.Field(i).Interface()
	}
	return m
}
