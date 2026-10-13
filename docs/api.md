# API

```{eval-rst}
.. module:: rctd
```

## Running RCTD

```{eval-rst}
.. autosummary::
   :toctree: generated

   run_rctd
   RCTD
   Reference
   RCTDConfig
```

## Results

```{eval-rst}
.. autosummary::
   :toctree: generated

   FullResult
   DoubletResult
   MultiResult
```

`SPOT_CLASS_NAMES` maps {attr}`DoubletResult.spot_class` codes to labels:
`["reject", "singlet", "doublet_certain", "doublet_uncertain"]`.

## Lower-level steps

The stages `run_rctd` chains together, for custom pipelines.

```{eval-rst}
.. autosummary::
   :toctree: generated

   fit_bulk
   choose_sigma
   run_full_mode
   run_doublet_mode
   run_multi_mode
```
