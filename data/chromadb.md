# ChromaDB
ChromaDB almacena vectores, documentos y metadatos. PersistentClient escribe los datos en una carpeta local para conservarlos entre reinicios. get permite buscar por identificador, mientras que query recupera vecinos por similitud vectorial. upsert inserta o actualiza un identificador y delete elimina registros. Debe usarse el mismo modelo de embeddings al indexar y consultar.
