import numpy as np
import pyedflib 


def readEDF(filename, prim_channels, ref_channel=None):
    """
    Read an EDF file from MASS Dataset, extract the desired primary channels and return them.
    Uses the reference channel to rereference them.
    """
    signals = None
    try:
        with pyedflib.EdfReader(filename) as edf_file:
            # Read the signal labels
            signal_labels = edf_file.getSignalLabels()
            # Extract desired primary channels
            indices_prim = [i for i, text in enumerate(signal_labels) for prim_channel in prim_channels if prim_channel in text]
            
            # Check that we have found the right primary channel (i.e. we have at least one)
            if len(indices_prim) < 1:
                raise AttributeError(f"No signals with labels {prim_channels} could be found in {filename}.")

            # Get all the desired signals
            signals = [edf_file.readSignal(index) for index in indices_prim]

            # If we have a referencing channel, use it to reference our signal
            if ref_channel is not None:
                # Extract its index
                index_ref = [i for i, text in enumerate(signal_labels) if ref_channel in text]

                # Check that we only have one ref channel given
                if len(index_ref) != 1:
                    raise AttributeError(f"Invalid reference channel {ref_channel} in {filename}.")

                # Load the reference signal and rereference all the previously loaded signals
                ref_signal = edf_file.readSignal(index_ref[0])
                assert len(ref) == len(signal), f"Issue with rereferencing signal in {filename}"
                signals = [signal - ref_signal for signal in signals]
    except OSError:
        print("File " + filename + " ignored because of corruption")

    return signals