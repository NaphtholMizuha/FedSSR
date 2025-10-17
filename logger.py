from loguru import logger

for level in logger._core.levels.values():
    print(f"{level.name:10} | {level.no:2} | {level.color}")