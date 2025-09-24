
## Command Line Configuration (Recommended)

**The application now supports command line arguments for configuration, which is the recommended approach.**

Use the `--classes` and `--territories` arguments when running the main script:

```bash
python run.py --classes road sidewalk tree --territories "Paris France:2" "London UK:1"
```

## Legacy CSV Configuration (Deprecated)

The CSV-based configuration is still supported for backward compatibility but is deprecated in favor of command line arguments.

Set the desired prompts at "configs/prompted_classes.csv", following the available model.
Set the desired territories at "configs/territories.csv", following the available model, territories must be geocodable in Nomitatim API, in case of more than one results, the one with biggest "relevance" will be chosen.  

In both cases watch for csv consistency (It's recommended to use CSVLint and RAinbowCSV).

## options.py

Here you must include the 

## prompted_classes.csv

Legacy CSV format for class prompts. Use command line `--classes` argument instead.

## territories.csv

Legacy CSV format for territories. Use command line `--territories` argument instead.