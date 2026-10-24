api.py contains the API endpoints for response generation, and is written using FastAPI.  
To run the web server and test API locally, from /backend use  
```uv run uvicorn api:app```

schemas.py contains the main schemas for generations, interventions, and responses.

model.py contains the model loading logic

generation.py contains the functions that call the model, generate text, and intervention wrappers

interventions.py contains the functions that perform interventions on the model, like attention head ablation, MLP unit ablation, etc.

test_schemas.py contains test for correctness of all schemas.  
test_generation.py contains tests for the basic generation functions, and for the custom stopping criteria functionality.  
test_interventions.py contains tests for the functions that modify the models.  
To run test session use  
```uv run pytest -v```

/JSON_examples contains sample JSON files that demonstrate generation requests of different kinds, and also successful and unsuccessful API responses.