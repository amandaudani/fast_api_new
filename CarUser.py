from pydantic import BaseModel # type: ignore

class CarUser(BaseModel):
  age: int
  gender: int