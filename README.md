# sandag-abm3-example

This example demonstrates the SANDAG ABM3 model.  This is ActivitySim's 
"prototypical" example of a two-zone model.

A full-scale exercise of the model is available via the `exercise.py` script,
which will download (large) data files to run the full-scale tests.

You can also run a smaller sized example set using the data files included
in this repository, using this command:

```
activitysim run -c configs/common -c configs/resident -d data -o output -s settings_mp.yaml --ext extensions
```
