# Pydantic
BaseModel define campos y tipos de un contrato de datos. Field permite declarar límites como ge, le y min_length. Un valor incompatible provoca ValidationError. model_validate transforma y valida datos de entrada. model_dump produce un diccionario y model_dump_json serializa el resultado. Validar la estructura no demuestra que el contenido generado por un modelo sea verdadero.
