FROM ollama/ollama:latest

ENV OLLAMA_HOST=0.0.0.0:11434

COPY dockerfiles/ollama/start.sh /usr/local/bin/start-ollama.sh
RUN chmod +x /usr/local/bin/start-ollama.sh

ENTRYPOINT ["/usr/local/bin/start-ollama.sh"]
