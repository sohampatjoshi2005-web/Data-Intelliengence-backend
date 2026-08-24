# Backend API Reference

Base URL: `http://localhost:8000`

Auth: If enabled, send headers `x-role` and `x-api-key`.
Roles used: `viewer`, `data_scientist`, `ml_engineer`, `risk_reviewer`, `approver`.

## Health & Models

1. `GET /health`
   - Returns backend status and provider availability.

2. `GET /models`
   - Returns model/provider configuration and auth mode.

## Connectors

3. `GET /connectors`
   - Lists supported connectors.

4. `POST /connectors/test`
   - Body: `{ connector, config }`
   - Role: `data_scientist`

5. `POST /connectors/load`
   - Body: `{ connector, config, query?, table?, limit }`
   - Role: `data_scientist`
   - Returns preview rows.

## Structured AutoML

6. `POST /orchestrate` (multipart)
   - Form fields: `file`, `business_problem`, `target_column`, `model_family`, `fixed_model`, `llm_provider`

7. `POST /orchestrate-from-connector`
   - Body: `{ connector, config, query?, table?, limit, dataset_name?, business_problem, target_column, model_family, fixed_model, llm_provider }`
   - Role: `data_scientist`

8. `POST /structured/preview` (multipart)
   - Form fields: `file`
   - Role: `viewer`
   - Returns: `{ columns, rows, shape }`

9. `POST /structured/tune`
   - Body: `{ rows, target, task, model_key, n_trials }`
   - Role: `data_scientist`
   - Returns: `{ baseline_score, tuned_score, improvement_abs, improvement_pct, best_params, trial_scores?, best_scores? }`

10. `POST /structured/predict/train`
    - Body: `{ rows, target, task, model_key }`
    - Role: `data_scientist`
    - Returns: `{ model_id, task, feature_columns }`

11. `POST /structured/predict`
    - Body: `{ model_id, rows, return_proba? }`
    - Role: `viewer`
    - Returns: `{ predictions, probabilities? }`

12. `POST /structured/online/start`
    - Body: `{ task }`
    - Role: `data_scientist`
    - Returns: `{ stream_id, task }`

13. `POST /structured/online/batch`
    - Body: `{ stream_id, rows, target, max_rows? }`
    - Role: `data_scientist`
    - Returns: `{ processed, drift_hits, drift_events, accuracy, running_variance?, history, accuracy_history, variance_history }`

14. `GET /structured/online/status`
    - Query params: `stream_id`
    - Role: `viewer`
    - Returns: `{ drift_events, accuracy, history, accuracy_history, variance_history }`

15. `POST /structured/online/stop`
    - Query params: `stream_id`
    - Role: `data_scientist`
    - Returns: `{ stream_id, ok }`

16. `POST /structured/explain`
    - Body: `{ result, business_problem?, drift_context?, response_style?, llm_provider? }`
    - Role: `viewer`
    - Returns: `{ explanation }`

17. `POST /structured/business-summary`
    - Body: `{ result, business_problem? }`
    - Role: `viewer`
    - Returns: `{ summary }`

18. `POST /structured/explainability`
    - Body: `{ rows, target, task, model_key, top_n? }`
    - Role: `data_scientist`
    - Returns: `{ explainability }`

## DeepEval

19. `POST /deepeval/run`
   - Body: `{ input_text, actual_output, context?, model_name?, metrics? }`

## Evaluation Tools

20. `GET /evaluation-tools`
   - Returns evaluation tool catalog.

## Orchestration

21. `GET /orchestration/status`
    - Returns orchestration/infra status.

## Model Registry

22. `GET /registry/models`
    - Role: `viewer`

23. `POST /registry/approve`
    - Body: `{ model_id, approver, note? }`
    - Role: `risk_reviewer`

24. `POST /registry/promote`
    - Body: `{ model_id, stage, actor }`
    - Role: `approver`

25. `POST /registry/rollback`
    - Body: `{ model_id, actor }`
    - Role: `approver`

## Feature Store

26. `GET /feature-store/tables`
    - Role: `viewer`

27. `POST /feature-store/offline/upsert`
    - Body: `{ table, rows }`
    - Role: `data_scientist`

28. `POST /feature-store/online/materialize`
    - Body: `{ table, key_col, ts_col? }`
    - Role: `ml_engineer`

29. `POST /feature-store/online/read`
    - Body: `{ table, key_col, key_val }`
    - Role: `viewer`

## Chat

30. `POST /chat`
    - Body: `{ question, context?, provider? }`

31. `POST /chat/structured-file` (multipart)
    - Form fields: `file`, `question`, `provider`
    - Returns: `{ answer }`

## Knowledge Base (Unstructured)

32. `POST /kb/build` (multipart)
    - Form fields: `file`, `dataset_id`, `llm_provider`, plus performance tuning fields.

33. `POST /kb/query`
    - Body: `{ dataset_id, query, top_k, llm_provider }`

## Knowledge Graph

34. `POST /kg/build`
    - Body: `{ dataset_id }`

35. `POST /kg/query`
    - Body: `{ dataset_id, entity, limit }`

36. `POST /kg/subgraph`
    - Body: `{ dataset_id, seed_entity, hops, limit }`

## Unstructured Analysis (Any File)

37. `POST /unstructured/analyze` (multipart)
    - Form fields: `file`
    - Returns: `{ file_name, file_type, text_preview, text_length, entities, entity_counts, warnings?, duration?, caption?, caption_model? }`
    - Image/video captioning uses **Ollama Vision** with model `OLLAMA_VISION_MODEL` (default: `moondream`).

## Transformations

38. `POST /transform/run` (multipart)
    - Form fields: `file`, `transform_type`
    - Role: `viewer`
    - Returns: `{ file_name, media_type, content_base64 }`

## EdgeQuake (Proxy)

39. `POST /edgequake/upload` (multipart)
    - Form fields: `file`, `base_url`
    - Role: `viewer`
    - Returns: `{ data: ... }`

40. `POST /edgequake/query`
    - Body: `{ base_url, query, mode }`
    - Role: `viewer`
    - Returns: `{ data: ... }`

## Analytics

41. `POST /analytics/query`
    - Body: `{ dataframe, query, chat_context?, llm_provider?, force_sql? }`

42. `POST /analytics/query-file` (multipart)
    - Form fields: `file`, `query`, `chat_context`, `llm_provider`, `force_sql`

43. `POST /analytics/insights`
    - Body: `{ dataframe, llm_provider }`

44. `POST /analytics/insights-file` (multipart)
    - Form fields: `file`, `llm_provider`

45. `POST /analytics/visuals-file` (multipart)
    - Form fields: `file`

46. `POST /analytics/run-file-stream` (multipart, streaming)
    - Form fields: `file`, `query`, `chat_context`, `llm_provider`
    - Streamed events: `status`, `reasoning_chunk`, `result`, `error`

47. `POST /analytics/run-sql`
    - Body: `{ dataframe, sql }`
    - Returns: `{ rows, columns, row_count }`

48. `POST /analytics/dashboard`
    - Body: `{ dataframe }`
    - Returns: `{ numeric_summary, categorical_summary, correlations }`

49. `POST /analytics/nl-to-sql`
    - Body: `{ dataframe, query, llm_provider? }`
    - Returns: `{ sql }`

50. `POST /analytics/sql-explain`
    - Body: `{ sql, llm_provider? }`
    - Returns: `{ explanation }`
