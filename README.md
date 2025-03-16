# CS 631: Epidemiological Network Model at Scale

## Install

```shell
pip install
```

## Run

To run an example simulation, call either the `parallel` or `serial` modules
directory:

```shell
python src/parallel.py
python src/serial.py
```

I will update the code shortly to save or pipe the output somewhere.
If you run either of the above modules, make sure to modify the
`if __name__ == "__main__"` section to save or catch the output.

## TODO

- Realtime visualization:
  - Add queue-based data updates and run data gathering in a thread in main to
    collect info and update visualization.
- Improve decomposition scripts
- Create interactive visualization app
