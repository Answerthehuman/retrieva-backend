# Source: https://milvus.io/docs/boolean.md

Filtering Explained | Milvus Documentation

[🚀 Zilliz Cloud: fully managed Milvus - 10x faster. Zero hassle. Built for AI. Try Free Now →](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_top_banner&utm_content=docs/boolean.md)

Milvus

<!-- image -->

Zilliz

<!-- image -->

- Why Milvus
- [Docs](\docs)
- Tutorials
- Tools
- [Blog](\blog)
- Community

[Star 42.6K](https://github.com/milvus-io/milvus) [Book a Demo](\contact) [Try Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_nav_right&utm_content=docs/boolean.md)

Milvus

<!-- image -->

Zilliz

<!-- image -->

[Docs](\docs) Tutorials Tools [Blog](\blog) Community [Book a Demo](\contact) [Try Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_nav_right&utm_content=docs/boolean.md)

Search [Home](\docs) [v2.6.x](\docs\boolean.md)

- About Milvus
- Get Started
- Concepts
- User Guide
- Data Import
- Administration Guide
- Tools
- Integrations
- Tutorials
- FAQs
- API Reference

- [Home](\)
- [Docs](\docs)
- User Guide
- Search
- Filtering
- Filtering Explained

Copy page

# Filtering Explained

Milvus provides powerful filtering capabilities that enable precise querying of your data. Filter expressions allow you to target specific scalar fields and refine search results with different conditions. This guide explains how to use filter expressions in Milvus, with examples focused on query operations. You can also apply these filters in search and delete requests.

## Basic operators

Milvus supports several basic operators for filtering data:

- **Comparison Operators** : `==` , `!=` , `>` , `<` , `>=` , and `<=` allow filtering based on numeric or text fields.
- **Range Filters** : `IN` and `LIKE` help match specific value ranges or sets.
- **Arithmetic Operators** : `+` , `-` , `*` , `/` , `%` , and `**` are used for calculations involving numeric fields.
- **Logical Operators** : `AND` , `OR` , and `NOT` combine multiple conditions into complex expressions.
- **IS NULL and IS NOT NULL Operators** : The `IS NULL` and `IS NOT NULL` operators are used to filter fields based on whether they contain a null value (absence of data). For details, refer to [Basic Operators](\docs\basic-operators.md#IS-NULL-and-IS-NOT-NULL-Operators) .

### Example: Filtering by Color

To find entities with primary colors (red, green, or blue) in a scalar field `color` , use the following filter expression:

```
filter = 'color in ["red", "green", "blue"]'
```

### Example: Filtering JSON Fields

Milvus allows referencing keys in JSON fields. For instance, if you have a JSON field `product` with keys `price` and `model` , and want to find products with a specific model and price lower than 1,850, use this filter expression:

```
filter = 'product["model"] == "JSN-087" AND product["price"] < 1850'
```

### Example: Filtering Array Fields

If you have an array field `history_temperatures` containing the records of average temperatures reported by observatories since the year 2000, and want to find observatories where the temperature in 2009 (the 10th recorded ) exceeds 23°C, use this expression:

```
filter = 'history_temperatures[10] > 23'
```

For more information on these basic operators, refer to [Basic Operators](\docs\basic-operators.md) .

## Filter expression templates

When filtering using CJK characters, processing can be more complex due to their larger character sets and encoding differences. This can result in slower performance, especially with the `IN` operator.

Milvus introduces filter expression templating to optimize performance when working with CJK characters. By separating dynamic values from the filter expression, the query engine handles parameter insertion more efficiently.

### Example

To find individuals over the age of 25 living in either "北京" (Beijing) or "上海" (Shanghai), use the following template expression:

```
filter = "age > 25 AND city IN ['北京', '上海']"
```

To improve performance, use this variation with parameters:

```
filter = "age > {age} AND city in {city}" ,
filter_params = { "age" : 25 , "city" : [ "北京" , "上海" ]}
```

This approach reduces parsing overhead and improves query speed. For more information, see [Filter Templating](\docs\filtering-templating.md) .

## Data type-specific operators

Milvus provides advanced filtering operators for specific data types, such as JSON, ARRAY, and VARCHAR fields.

### JSON field-specific operators

Milvus offers advanced operators for querying JSON fields, enabling precise filtering within complex JSON structures:

`JSON_CONTAINS(identifier, jsonExpr)` : Checks if a JSON expression exists in the field.

```
# JSON data: {"tags": ["electronics", "sale", "new"]}
filter = 'json_contains(tags, "sale")'
```

`JSON_CONTAINS_ALL(identifier, jsonExpr)` : Ensures all elements of the JSON expression are present.

```
# JSON data: {"tags": ["electronics", "sale", "new", "discount"]}
filter = 'json_contains_all(tags, ["electronics", "sale", "new"])'
```

`JSON_CONTAINS_ANY(identifier, jsonExpr)` : Filters for entities where at least one element exists in the JSON expression.

```
# JSON data: {"tags": ["electronics", "sale", "new"]}
filter = 'json_contains_any(tags, ["electronics", "new", "clearance"])'
```

For more details on JSON operators, refer to [JSON Operators](\docs\json-operators.md) .

### ARRAY field-specific operators

Milvus provides advanced filtering operators for array fields, such as `ARRAY_CONTAINS` , `ARRAY_CONTAINS_ALL` , `ARRAY_CONTAINS_ANY` , and `ARRAY_LENGTH` , which allow fine-grained control over array data:

`ARRAY_CONTAINS` : Filters entities containing a specific element.

```
filter = "ARRAY_CONTAINS(history_temperatures, 23)"
```

`ARRAY_CONTAINS_ALL` : Filters entities where all elements in a list are present.

```
filter = "ARRAY_CONTAINS_ALL(history_temperatures, [23, 24])"
```

`ARRAY_CONTAINS_ANY` : Filters entities containing any element from the list.

```
filter = "ARRAY_CONTAINS_ANY(history_temperatures, [23, 24])"
```

`ARRAY_LENGTH` : Filters based on the length of the array.

```
filter = "ARRAY_LENGTH(history_temperatures) < 10"
```

For more details on array operators, see [ARRAY Operators](\docs\array-operators.md) .

### VARCHAR field-specific operators

Milvus provides specialized operators for precise text-based searches on VARCHAR fields:

#### TEXT\_MATCH operator

The `TEXT_MATCH` operator allows precise document retrieval based on specific query terms. It is particularly useful for filtered searches that combine scalar filters with vector similarity searches. Unlike semantic searches, Text Match focuses on exact term occurrences.

Milvus uses Tantivy to support inverted indexing and term-based text search. The process involves:

1. **Analyzer** : Tokenizes and processes input text.
2. **Indexing** : Creates an inverted index mapping unique tokens to documents.

For more details, refer to [Text Match](\docs\keyword-match.md) .

#### PHRASE\_MATCH operator Compatible with Milvus 2.6.x

The **PHRASE\_MATCH** operator enables precise retrieval of documents based on exact phrase matches, considering both the order and adjacency of query terms.

For more details, refer to [Phrase Match](\docs\phrase-match.md) .

##### Table of contents

- [Filtering Explained](#Filtering-Explained)
- [Basic operators](#Basic-operators)
- [Example: Filtering by Color](#Example-Filtering-by-Color)
- [Example: Filtering JSON Fields](#Example-Filtering-JSON-Fields)
- [Example: Filtering Array Fields](#Example-Filtering-Array-Fields)
- [Filter expression templates](#Filter-expression-templates)
- [Example](#Example)
- [Data type-specific operators](#Data-type-specific-operators)
- [JSON field-specific operators](#JSON-field-specific-operators)
- [ARRAY field-specific operators](#ARRAY-field-specific-operators)
- [VARCHAR field-specific operators](#VARCHAR-field-specific-operators)

## Try Managed Milvus for Free

Zilliz Cloud is hassle-free, powered by Milvus and 10x faster.

[Get Started](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_right_card&utm_content=docs/boolean.md)

- [Edit this page](https://github.com/milvus-io/milvus-docs/edit/v2.6.x/site/en/userGuide/search-query-get/boolean/boolean.md)
- [Create an issue](https://github.com/milvus-io/milvus-docs/issues/new/choose)

##### Feedback

Was this page helpful?

### Get Milvus Updates

Subscribe

Copyright © Milvus. 2026 All rights reserved.

### Resources

- [Docs](\docs)
- [Blog](\blog)
- [Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_footer&utm_content=docs/boolean.md)
- [Book a Demo](\contact)
- [AI Quick Reference](\ai-quick-reference)

### Tutorials

- [Bootcamps](\bootcamp)
- [Demo](\milvus-demos)
- [Video](https://www.youtube.com/c/MilvusVectorDatabase)

### Tools

- [Attu](https://github.com/zilliztech/attu)
- [Milvus CLI](https://github.com/zilliztech/milvus_cli)
- [Milvus Sizing Tool](\tools\sizing)
- [Milvus Backup Tool](https://github.com/zilliztech/milvus-backup)
- [Vector Transport Service (VTS)](https://github.com/zilliztech/vts)
- [Deep Searcher](https://github.com/zilliztech/deep-searcher)
- [Claude Context](https://github.com/zilliztech/claude-context)

### Community

- [Milvus Office Hours](https://meetings.hubspot.com/chloe-williams1/milvus-office-hour?uuid=4cb203e5-482a-47e0-90a6-7acc511d61f4)
- [Slack](https://milvus.io/slack)
- [Discord](https://milvus.io/discord)
- [Github](https://github.com/milvus-io/milvus)

Ask AI

<!-- image -->

---

# Source: https://milvus.io/docs/basic-operators.md

Basic Operators | Milvus Documentation

[🚀 Zilliz Cloud: fully managed Milvus - 10x faster. Zero hassle. Built for AI. Try Free Now →](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_top_banner&utm_content=docs/basic-operators.md)

Milvus

<!-- image -->

Zilliz

<!-- image -->

- Why Milvus
- [Docs](\docs)
- Tutorials
- Tools
- [Blog](\blog)
- Community

[Star 42.6K](https://github.com/milvus-io/milvus) [Book a Demo](\contact) [Try Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_nav_right&utm_content=docs/basic-operators.md)

Milvus

<!-- image -->

Zilliz

<!-- image -->

[Docs](\docs) Tutorials Tools [Blog](\blog) Community [Book a Demo](\contact) [Try Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_nav_right&utm_content=docs/basic-operators.md)

Search [Home](\docs) [v2.6.x](\docs\basic-operators.md)

- About Milvus
- Get Started
- Concepts
- User Guide
- Data Import
- Administration Guide
- Tools
- Integrations
- Tutorials
- FAQs
- API Reference

- [Home](\)
- [Docs](\docs)
- User Guide
- Search
- Filtering
- Basic Operators

Copy page

# Basic Operators

Milvus provides a rich set of basic operators to help you filter and query data efficiently. These operators allow you to refine your search conditions based on scalar fields, numeric calculations, logical conditions, and more. Understanding how to use these operators is crucial for building precise queries and maximizing the efficiency of your searches.

## Comparison operators

Comparison operators are used to filter data based on equality, inequality, or size. They are applicable to numeric and text fields.

### Supported Comparison Operators:

- `==` (Equal to)
- `!=` (Not equal to)
- `>` (Greater than)
- `<` (Less than)
- `>=` (Greater than or equal to)
- `<=` (Less than or equal to)

### Example 1: Filtering with Equal To ( == )

Assume you have a field named `status` and you want to find all entities where `status` is "active". You can use the equality operator `==` :

```
filter = 'status == "active"'
```

### Example 2: Filtering with Not Equal To ( != )

To find entities where `status` is not "inactive":

```
filter = 'status != "inactive"'
```

### Example 3: Filtering with Greater Than ( &gt; )

If you want to find all entities with an `age` greater than 30:

```
filter = 'age > 30'
```

### Example 4: Filtering with Less Than

To find entities where `price` is less than 100:

```
filter = 'price < 100'
```

### Example 5: Filtering with Greater Than or Equal To ( &gt;= )

If you want to find all entities with `rating` greater than or equal to 4:

```
filter = 'rating >= 4'
```

### Example 6: Filtering with Less Than or Equal To

To find entities with `discount` less than or equal to 10%:

```
filter = 'discount <= 10'
```

## Range operators

Range operators help filter data based on specific sets or ranges of values.

### Supported Range Operators:

- `IN` : Used to match values within a specific set or range.
- `LIKE` : Used to match a pattern (mostly for text fields). Milvus allows you to build an `NGRAM` index on VARCHAR or JSON fields to accelerate text queries. For details, refer to [NGRAM](\docs\ngram.md) .

### Example 1: Using IN to Match Multiple Values

If you want to find all entities where the `color` is either "red", "green", or "blue":

```
filter = 'color in ["red", "green", "blue"]'
```

This is useful when you want to check for membership in a list of values.

### Example 2: Using LIKE for Pattern Matching

The `LIKE` operator is used for pattern matching in string fields. It can match substrings in different positions within the text: as a **prefix** , **infix** , or **suffix** . The `LIKE` operator uses the `%` symbol as a wildcard, which can match any number of characters (including zero).

In most cases, **infix** or **suffix** matching is significantly slower than prefix matching. Use them with caution if performance is critical.

### Prefix Match (Starts With)

To perform a **prefix** match, where the string starts with a given pattern, you can place the pattern at the beginning and use `%` to match any characters following it. For example, to find all products whose `name` starts with "Prod":

```
filter = 'name LIKE "Prod%"'
```

This will match any product whose name starts with "Prod", such as "Product A", "Product B", etc.

### Suffix Match (Ends With)

For a **suffix** match, where the string ends with a given pattern, place the `%` symbol at the beginning of the pattern. For example, to find all products whose `name` ends with "XYZ":

```
filter = 'name LIKE "%XYZ"'
```

This will match any product whose name ends with "XYZ", such as "ProductXYZ", "SampleXYZ", etc.

### Infix Match (Contains)

To perform an **infix** match, where the pattern can appear anywhere in the string, you can place the `%` symbol at both the beginning and the end of the pattern. For example, to find all products whose `name` contains the word "Pro":

```
filter = 'name LIKE "%Pro%"'
```

This will match any product whose name contains the substring "Pro", such as "Product", "ProLine", or "SuperPro".

## Arithmetic Operators

Arithmetic operators allow you to create conditions based on calculations involving numeric fields.

### Supported Arithmetic Operators:

- `+` (Addition)
- `-` (Subtraction)
- `*` (Multiplication)
- `/` (Division)
- `%` (Modulus)
- `**` (Exponentiation)

### Example 1: Using Modulus ( % )

To find entities where the `id` is an even number (i.e., divisible by 2):

```
filter = 'id % 2 == 0'
```

### Example 2: Using Exponentiation ( ** )

To find entities where `price` raised to the power of 2 is greater than 1000:

```
filter = 'price ** 2 > 1000'
```

## Logical Operators

Logical operators are used to combine multiple conditions into a more complex filter expression. These include `AND` , `OR` , and `NOT` .

### Supported Logical Operators:

- `AND` : Combines multiple conditions that must all be true.
- `OR` : Combines conditions where at least one must be true.
- `NOT` : Negates a condition.

### Example 1: Using AND to Combine Conditions

To find all products where `price` is greater than 100 and `stock` is greater than 50:

```
filter = 'price > 100 AND stock > 50'
```

### Example 2: Using OR to Combine Conditions

To find all products where `color` is either "red" or "blue":

```
filter = 'color == "red" OR color == "blue"'
```

### Example 3: Using NOT to Exclude a Condition

To find all products where `color` is not "green":

```
filter = 'NOT color == "green"'
```

## IS NULL and IS NOT NULL Operators

The `IS NULL` and `IS NOT NULL` operators are used to filter fields based on whether they contain a null value (absence of data).

- `IS NULL` : Identifies entities where a specific field contains a null value, i.e., the value is absent or undefined.
- `IS NOT NULL` : Identifies entities where a specific field contains any value other than null, meaning the field has a valid, defined value.

The operators are case-insensitive, so you can use `IS NULL` or `is null` , and `IS NOT NULL` or `is not null` .

### Regular Scalar Fields with Null Values

Milvus allows filtering on regular scalar fields, such as strings or numbers, with null values.

An empty string `""` is not treated as a null value for a `VARCHAR` field.

To retrieve entities where the `description` field is null:

```
filter = 'description IS NULL'
```

To retrieve entities where the `description` field is not null:

```
filter = 'description IS NOT NULL'
```

To retrieve entities where the `description` field is not null and the `price` field is higher than 10:

```
filter = 'description IS NOT NULL AND price > 10'
```

### JSON Fields with Null Values

Milvus allows filtering on JSON fields that contain null values. A JSON field is treated as null in the following ways:

- The entire JSON object is explicitly set to None (null), for example, `{"metadata": None}` .
- The JSON field itself is completely missing from the entity.

If some elements within a JSON object are null (e.g. individual keys), the field is still considered non-null. For example, `\{"metadata": \{"category": None, "price": 99.99}}` is not treated as null, even though the `category` key is null.

To further illustrate how Milvus handles JSON fields with null values, consider the following sample data with a JSON field `metadata` :

```
data = [
  { "metadata" : { "category" : "electronics" , "price" : 99.99 , "brand" : "BrandA" }, "pk" : 1 , "embedding" : [ 0.12 , 0.34 , 0.56 ]
  },
  { "metadata" : None , # Entire JSON object is null "pk" : 2 , "embedding" : [ 0.56 , 0.78 , 0.90 ]
  },
  { # JSON field `metadata` is completely missing "pk" : 3 , "embedding" : [ 0.91 , 0.18 , 0.23 ]
  },
  { "metadata" : { "category" : None , "price" : 99.99 , "brand" : "BrandA" }, # Individual key value is null "pk" : 4 , "embedding" : [ 0.56 , 0.38 , 0.21 ]
  }
]
```

**Example 1: Retrieve entities where metadata is null**

To find entities where the `metadata` field is either missing or explicitly set to None:

`filter = 'metadata IS NULL'
# Example output: # data: [ #     "{'metadata': None, 'pk': 2}", #     "{'metadata': None, 'pk': 3}" # ]` **`Example 2: Retrieve entities where metadata is not null`** `To find entities where the metadata field is not null:
filter = 'metadata IS NOT NULL'
# Example output: # data: [ #     "{'metadata': {'category': 'electronics', 'price': 99.99, 'brand': 'BrandA'}, 'pk': 1}", #     "{'metadata': {'category': None, 'price': 99.99, 'brand': 'BrandA'}, 'pk': 4}" # ]
ARRAY Fields with Null Values Milvus allows filtering on ARRAY fields that contain null values. An ARRAY field is treated as null in the following ways:

The entire ARRAY field is explicitly set to None (null), for example, "tags": None .
The ARRAY field is completely missing from the entity.


An ARRAY field cannot contain partial null values as all elements in an ARRAY field must have the same data type. For details, refer to` [`Array Field`](\docs\array_data_type.md) `.

To further illustrate how Milvus handles ARRAY fields with null values, consider the following sample data with an ARRAY field tags :
data = [
  { "tags" : [ "pop" , "rock" , "classic" ], "ratings" : [ 5 , 4 , 3 ], "pk" : 1 , "embedding" : [ 0.12 , 0.34 , 0.56 ]
  },
  { "tags" : None , # Entire ARRAY is null "ratings" : [ 4 , 5 ], "pk" : 2 , "embedding" : [ 0.78 , 0.91 , 0.23 ]
  },
  { # The tags field is completely missing "ratings" : [ 9 , 5 ], "pk" : 3 , "embedding" : [ 0.18 , 0.11 , 0.23 ]
  }
]` **`Example 1: Retrieve entities where tags is null`** `To retrieve entities where the tags field is either missing or explicitly set to None :
filter = 'tags IS NULL'
# Example output: # data: [ #     "{'tags': None, 'ratings': [4, 5], 'embedding': [0.78, 0.91, 0.23], 'pk': 2}", #     "{'tags': None, 'ratings': [9, 5], 'embedding': [0.18, 0.11, 0.23], 'pk': 3}" # ]` **`Example 2: Retrieve entities where tags is not null`** `To retrieve entities where the tags field is not null:
filter = 'tags IS NOT NULL'
# Example output: # data: [ #     "{'metadata': {'category': 'electronics', 'price': 99.99, 'brand': 'BrandA'}, 'pk': 1}", #     "{'metadata': {'category': None, 'price': 99.99, 'brand': 'BrandA'}, 'pk': 4}" # ]
Tips on Using Basic Operators with JSON and ARRAY Fields While the basic operators in Milvus are versatile and can be applied to scalar fields, they can also be effectively used with the keys and indexes in the JSON and ARRAY fields.
For example, if you have a product field that contains multiple keys like price , model , and tags , always reference the key directly:
filter = 'product["price"] > 1000'

To find records where the first temperature in an array of recorded temperatures exceeds a certain value, use:
filter = 'history_temperatures[0] > 30'

Conclusion Milvus offers a range of basic operators that give you flexibility in filtering and querying your data. By combining comparison, range, arithmetic, and logical operators, you can create powerful filter expressions to narrow down your search results and retrieve the data you need efficiently.
FAQ` **`Is there a limit to the length of the match value list in filter conditions (e.g., filter='color in ["red", "green", "blue"]')? What should I do if the list is too long?`** `Zilliz Cloud does not impose a length limit on the match value list in filter conditions. However, an excessively long list can significantly impact query performance.
If your filter condition includes a long list of match values or a complex expression with many elements, we recommend using` [`Filter Templating`](\docs\filtering-templating.md) `to improve query performance.`

##### Table of contents

- [Basic Operators](#Basic-Operators)
- [Comparison operators](#Comparison-operators)
- [Supported Comparison Operators:](#Supported-Comparison-Operators)
- [Example 1: Filtering with Equal To (==)](#Example-1-Filtering-with-Equal-To-)
- [Example 2: Filtering with Not Equal To (!=)](#Example-2-Filtering-with-Not-Equal-To-)
- [Example 3: Filtering with Greater Than (&gt;)](#Example-3-Filtering-with-Greater-Than-)
- [Example 4: Filtering with Less Than](#Example-4-Filtering-with-Less-Than)
- [Example 5: Filtering with Greater Than or Equal To (&gt;=)](#Example-5-Filtering-with-Greater-Than-or-Equal-To-)
- [Example 6: Filtering with Less Than or Equal To](#Example-6-Filtering-with-Less-Than-or-Equal-To)
- [Range operators](#Range-operators)
- [Supported Range Operators:](#Supported-Range-Operators)
- [Example 1: Using IN to Match Multiple Values](#Example-1-Using-IN-to-Match-Multiple-Values)
- [Example 2: Using LIKE for Pattern Matching](#Example-2-Using-LIKE-for-Pattern-Matching)
- [Prefix Match (Starts With)](#Prefix-Match-Starts-With)
- [Suffix Match (Ends With)](#Suffix-Match-Ends-With)
- [Infix Match (Contains)](#Infix-Match-Contains)
- [Arithmetic Operators](#Arithmetic-Operators)
- [Supported Arithmetic Operators:](#Supported-Arithmetic-Operators)
- [Example 1: Using Modulus (%)](#Example-1-Using-Modulus-)
- [Example 2: Using Exponentiation ()](#Example-2-Using-Exponentiation-)
- [Logical Operators](#Logical-Operators)
- [Supported Logical Operators:](#Supported-Logical-Operators)
- [Example 1: Using AND to Combine Conditions](#Example-1-Using-AND-to-Combine-Conditions)
- [Example 2: Using OR to Combine Conditions](#Example-2-Using-OR-to-Combine-Conditions)
- [Example 3: Using NOT to Exclude a Condition](#Example-3-Using-NOT-to-Exclude-a-Condition)
- [IS NULL and IS NOT NULL Operators](#IS-NULL-and-IS-NOT-NULL-Operators)
- [Regular Scalar Fields with Null Values](#Regular-Scalar-Fields-with-Null-Values)
- [JSON Fields with Null Values](#JSON-Fields-with-Null-Values)
- [ARRAY Fields with Null Values](#ARRAY-Fields-with-Null-Values)
- [Tips on Using Basic Operators with JSON and ARRAY Fields](#Tips-on-Using-Basic-Operators-with-JSON-and-ARRAY-Fields)
- [Conclusion](#Conclusion)
- [FAQ](#FAQ)

## Try Managed Milvus for Free

Zilliz Cloud is hassle-free, powered by Milvus and 10x faster.

[Get Started](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_right_card&utm_content=docs/basic-operators.md)

- [Edit this page](https://github.com/milvus-io/milvus-docs/edit/v2.6.x/site/en/userGuide/search-query-get/boolean/basic-operators.md)
- [Create an issue](https://github.com/milvus-io/milvus-docs/issues/new/choose)

##### Feedback

Was this page helpful?

### Get Milvus Updates

Subscribe

Copyright © Milvus. 2026 All rights reserved.

### Resources

- [Docs](\docs)
- [Blog](\blog)
- [Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_footer&utm_content=docs/basic-operators.md)
- [Book a Demo](\contact)
- [AI Quick Reference](\ai-quick-reference)

### Tutorials

- [Bootcamps](\bootcamp)
- [Demo](\milvus-demos)
- [Video](https://www.youtube.com/c/MilvusVectorDatabase)

### Tools

- [Attu](https://github.com/zilliztech/attu)
- [Milvus CLI](https://github.com/zilliztech/milvus_cli)
- [Milvus Sizing Tool](\tools\sizing)
- [Milvus Backup Tool](https://github.com/zilliztech/milvus-backup)
- [Vector Transport Service (VTS)](https://github.com/zilliztech/vts)
- [Deep Searcher](https://github.com/zilliztech/deep-searcher)
- [Claude Context](https://github.com/zilliztech/claude-context)

### Community

- [Milvus Office Hours](https://meetings.hubspot.com/chloe-williams1/milvus-office-hour?uuid=4cb203e5-482a-47e0-90a6-7acc511d61f4)
- [Slack](https://milvus.io/slack)
- [Discord](https://milvus.io/discord)
- [Github](https://github.com/milvus-io/milvus)

Ask AI

<!-- image -->

---

# Source: https://milvus.io/docs/filtering-templating.md

Filter Templating | Milvus Documentation

[🚀 Zilliz Cloud: fully managed Milvus - 10x faster. Zero hassle. Built for AI. Try Free Now →](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_top_banner&utm_content=docs/filtering-templating.md)

Milvus

<!-- image -->

Zilliz

<!-- image -->

- Why Milvus
- [Docs](\docs)
- Tutorials
- Tools
- [Blog](\blog)
- Community

[Star 42.6K](https://github.com/milvus-io/milvus) [Book a Demo](\contact) [Try Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_nav_right&utm_content=docs/filtering-templating.md)

Milvus

<!-- image -->

Zilliz

<!-- image -->

[Docs](\docs) Tutorials Tools [Blog](\blog) Community [Book a Demo](\contact) [Try Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_nav_right&utm_content=docs/filtering-templating.md)

Search [Home](\docs) [v2.6.x](\docs\filtering-templating.md)

- About Milvus
- Get Started
- Concepts
- User Guide
- Data Import
- Administration Guide
- Tools
- Integrations
- Tutorials
- FAQs
- API Reference

- [Home](\)
- [Docs](\docs)
- User Guide
- Search
- Filtering
- Filtering Templating

Copy page

# Filter Templating

In Milvus, complex filter expressions with numerous elements, especially those involving non-ASCII characters like CJK characters, can significantly affect query performance. To address this, Milvus introduces a filter expression templating mechanism designed to improve efficiency by reducing the time spent parsing complex expressions. This page explains using filter expression templating in search, query, and delete operations.

## Overview

Filter expression templating allows you to create filter expressions with placeholders, which can be dynamically substituted with values during query execution. Using templating, you avoid embedding large arrays or complex expressions directly into the filter, reducing parsing time and improving query performance.

Let's say you have a filter expression involving two fields, `age` and `city` , and you want to find all people whose age is greater than 25 and who live in either "北京" (Beijing) or "上海" (Shanghai). Instead of directly embedding the values in the filter expression, you can use a template:

```
filter = "age > {age} AND city IN {city}" filter_params = { "age" : 25 , "city" : [ "北京" , "上海" ]}
```

Here, `{age}` and `{city}` are placeholders that will be replaced with the actual values in `filter_params` when the query is executed.

Using filter expression templating in Milvus has several key advantages:

- **Reduced Parsing Time** : By replacing large or complex filter expressions with placeholders, the system spends less time parsing and processing the filter.
- **Improved Query Performance** : With reduced parsing overhead, query performance improves, leading to higher QPS and faster response times.
- **Scalability** : As your datasets grow and filter expressions become more complex, templating ensures that performance remains efficient and scalable.

## Search Operations

For search operations in Milvus, the `filter` expression is used to define the filtering condition, and the `filter_params` parameter is used to specify the values for the placeholders. The `filter_params` dictionary contains the dynamic values that Milvus will use to substitute into the filter expression.

```
expr = "age > {age} AND city IN {city}" filter_params = { "age" : 25 , "city" : [ "北京" , "上海" ]}
res = client.search( "hello_milvus" ,
    vectors[:nq], filter =expr,
    limit= 10 ,
    output_fields=[ "age" , "city" ],
    search_params={ "metric_type" : "COSINE" , "params" : { "search_list" : 100 }},
    filter_params=filter_params,
)
```

In this example, Milvus will dynamically replace `{age}` with `25` and `{city}` with `["北京", "上海"]` when executing the search.

## Query Operations

The same templating mechanism can be applied to query operations in Milvus. In the `query` function, you define the filter expression and use the `filter_params` to specify the values to substitute.

```
expr = "age > {age} AND city IN {city}" filter_params = { "age" : 25 , "city" : [ "北京" , "上海" ]}
res = client.query( "hello_milvus" , filter =expr,
    output_fields=[ "age" , "city" ],
    filter_params=filter_params
)
```

By using `filter_params` , Milvus efficiently handles the dynamic insertion of values, improving the speed of query execution.

## Delete Operations

You can also use filter expression templating in delete operations. Similar to search and query, the `filter` expression defines the conditions, and the `filter_params` provides the dynamic values for the placeholders.

```
expr = "age > {age} AND city IN {city}" filter_params = { "age" : 25 , "city" : [ "北京" , "上海" ]}
res = client.delete( "hello_milvus" , filter =expr,
    filter_params=filter_params
)
```

This approach improves the performance of delete operations, especially when dealing with complex filter conditions.

## Conclusion

Filter expression templating is an essential tool for optimizing query performance in Milvus. By using placeholders and the `filter_params` dictionary, you can significantly reduce the time spent parsing complex filter expressions. This leads to faster query execution and better overall performance.

##### Table of contents

- [Filter Templating](#Filter-Templating)
- [Overview](#Overview)
- [Search Operations](#Search-Operations)
- [Query Operations](#Query-Operations)
- [Delete Operations](#Delete-Operations)
- [Conclusion](#Conclusion)

## Try Managed Milvus for Free

Zilliz Cloud is hassle-free, powered by Milvus and 10x faster.

[Get Started](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_right_card&utm_content=docs/filtering-templating.md)

- [Edit this page](https://github.com/milvus-io/milvus-docs/edit/v2.6.x/site/en/userGuide/search-query-get/boolean/filtering-templating.md)
- [Create an issue](https://github.com/milvus-io/milvus-docs/issues/new/choose)

##### Feedback

Was this page helpful?

### Get Milvus Updates

Subscribe

Copyright © Milvus. 2026 All rights reserved.

### Resources

- [Docs](\docs)
- [Blog](\blog)
- [Managed Milvus](https://cloud.zilliz.com/signup?utm_source=milvusio&utm_medium=referral&utm_campaign=milvus_footer&utm_content=docs/filtering-templating.md)
- [Book a Demo](\contact)
- [AI Quick Reference](\ai-quick-reference)

### Tutorials

- [Bootcamps](\bootcamp)
- [Demo](\milvus-demos)
- [Video](https://www.youtube.com/c/MilvusVectorDatabase)

### Tools

- [Attu](https://github.com/zilliztech/attu)
- [Milvus CLI](https://github.com/zilliztech/milvus_cli)
- [Milvus Sizing Tool](\tools\sizing)
- [Milvus Backup Tool](https://github.com/zilliztech/milvus-backup)
- [Vector Transport Service (VTS)](https://github.com/zilliztech/vts)
- [Deep Searcher](https://github.com/zilliztech/deep-searcher)
- [Claude Context](https://github.com/zilliztech/claude-context)

### Community

- [Milvus Office Hours](https://meetings.hubspot.com/chloe-williams1/milvus-office-hour?uuid=4cb203e5-482a-47e0-90a6-7acc511d61f4)
- [Slack](https://milvus.io/slack)
- [Discord](https://milvus.io/discord)
- [Github](https://github.com/milvus-io/milvus)

Ask AI

<!-- image -->

---

