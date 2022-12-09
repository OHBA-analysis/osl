
Example scripts MaxFiltering data collected at Oxford
-----------------------------------------------------

`maxfilter.py` is an example script for maxfiltering data collected at Oxford after 2020. This can be executed with:

```
python maxfilter.py
```

There is a `--scanner` argument we pass to the maxfiltering function (line 22 in `maxfilter.py`). Depending on when the data was collected you will want to pass a different value for this argument:
- If the data was collected before December 2017, you want to pass `--scanner VectorView` to the maxfilter command. By default (if you don't pass an argument it will assume the scanner is VectorView).
- If the data was collected in 2017-2019 (before the Neo upgrade), you want to pass `--scanner VectorView2`.
- If the data was collected after 2019, you want to pass `--scanner Neo`.

`check_scanner.py` can be used to check what scanner was used. This can be executed with:

```
python check_scanner.py
```

If "TRIUX system at OHBA-3143" is in the scanner name use `--scanner Neo`. Otherwise, if "VectorView" is in the scanner name, you need `--scanner VectorView` or `--scanner VectorView2` depending on the date.
