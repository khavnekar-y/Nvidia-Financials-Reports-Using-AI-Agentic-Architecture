.PHONY: install prerequisites backend frontend rag docker-up docker-down
# Install project dependencies using Poetry 
install:
	poetry install

lock:
	poetry lock
update:
	poetry update




	
backend:
	poetry run uvicorn Backend.api:app --host 0.0.0.0 --port 8000 --reload
frontend:
	poetry run streamlit run .\Frontend\app.py --server.port 8501
pinecone:
	poetry run python .\Rag_modelings\rag_pinecone.py
pinecone-test:
	poetry run python .\Rag_modelings\Chunking_Stats\pinecone\pinecone_.py


llm:
	poetry run python .\Backend\litellm_query_generator.py



#Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down
clean:
	docker-compose down --volumes --rmi all --remove-orphans

# testing code
fernet:
	poetry run python .\testing_code\fernet_key_generation_key.py

mistraltest:
	poetry run python .\testing_code\mistral_test.py

# parsing methods
docling:
	poetry run python .\parsing_methods\doclingparsing.py

mistral:
	poetry run python .\parsing_methods\mistral_userpdf.py


# SerAPI Web API
NvidiaWebAgent:
	@echo "Running NVIDIA Web Search Agent..."
	poetry run python agents/websearch_agent.py

webagent-news:
	@echo "Running NVIDIA News Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_nvidia_news(days_ago=7); print(json.dumps(results, indent=2))"

webagent-financial:
	@echo "Running NVIDIA Financial Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_nvidia_financial(); print(json.dumps(results, indent=2))"

webagent-quarterly:
	@echo "Running NVIDIA Quarterly Report Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_quarterly_report_info(year=2023, quarter=4); print(json.dumps(results, indent=2))"

webagent-general:
	@echo "Running NVIDIA General Information Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_nvidia_general(); print(json.dumps(results, indent=2))"

webagent-report:
	@echo "Generating comprehensive NVIDIA research report..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import datetime; agent = NvidiaWebSearchAgent(); current_year = datetime.datetime.now().year; current_quarter = (datetime.datetime.now().month - 1) // 3 + 1; report = agent.generate_research_report(year=current_year, quarter=current_quarter); print(f'Report saved to: {report.get(\"saved_to\", \"unknown path\")}');"