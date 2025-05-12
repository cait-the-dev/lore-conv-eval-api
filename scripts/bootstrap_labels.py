from labeling_pipeline.heuristic import run as h
from labeling_pipeline.llm import run as l
from labeling_pipeline.merge import run as m

h()
l(max_calls=300)
m()
