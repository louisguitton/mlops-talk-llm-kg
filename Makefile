argilla:
	docker run -d \
		--name argilla \
		-p 6900:6900 \
		argilla/argilla-quickstart:latest

dbpedia:
	docker run -ti \
		--restart unless-stopped \
		--name dbpedia-spotlight.en \
		--mount source=spotlight-models,target=/opt/spotlight \
		-p 2222:80 \
		dbpedia/dbpedia-spotlight \
		spotlight.sh en

llm:
	ollama serve

llm-ui:
	docker run \
		-p 8080:8080 \
		--add-host=host.docker.internal:host-gateway \
		-v ollama-webui:/app/backend/data \
		--name ollama-webui \
		ghcr.io/ollama-webui/ollama-webui:main
