import numpy as np


def ResponseStats(spikeTrains, stimStart=10, stimDuration=50):
    """ Average spike rate during the stimulus response.
    :param spikeTrains, pandas.DataFrame of spike times for each cycle
    :type spikeTrains: pandas.DataFrame
    :param stimStart: Beginning of stimulus time
    :type stimStart: int
    :param stimDuration: Duration of stimulus response
    :type stimDuration: int
    :returns: responseStats, list: average and standard deviation of the rate during the stimulus response
    """
    dur = 0.001 * stimDuration
    responseSpikeCount = []
    spontSpikeCount = []
    for k in spikeTrains.keys():
        spk = spikeTrains[k]
        responseSpikeCount.append(len(spk[spk < stimStart + stimDuration + 10]) / dur)
        spontSpikeCount.append(len(spk[spk > 100]) / 0.1)

    if len(responseSpikeCount) > 0:
        responseStats = [np.mean(responseSpikeCount), np.std(responseSpikeCount)]
    else:
        responseStats = [0, 0]
    if len(spontSpikeCount) > 0:
        spontStats = [np.mean(spontSpikeCount), np.std(spontSpikeCount)]
    else:
        spontStats = [0, 0]

    return responseStats


def PresentationStats(spikeTrains):
    """ Average spike rate during the stimulus response.
    :param spikeTrains, pandas.DataFrame of spike times for each cycle
    :type spikeTrains: pandas.DataFrame
    :param stimStart: Beginning of stimulus time
    :type stimStart: int
    :param stimDuration: Duration of stimulus response
    :type stimDuration: int
    :returns: responseStats, list: average and standard deviation of the rate during the stimulus response
    """
    responseSpikeCount = []
    for k in spikeTrains.keys():
        spk = spikeTrains[k]

        spikes = spk.values
        spikes = np.array(spikes)
        spikes = spikes[~np.isnan(spikes)]
        responseSpikeCount.append(len(spikes))

    if len(responseSpikeCount) > 0:
        responseStats = [np.mean(responseSpikeCount), np.std(responseSpikeCount)]
    else:
        responseStats = [0, 0]

    return responseStats


def ResponseStatsSpikes(spikeTrains, stimStart=10, stimDuration=50):
    """ Average spike rate during the stimulus response.
    :param spikeTrains, pandas.DataFrame of spike times for each cycle
    :type spikeTrains: pandas.DataFrame
    :param stimStart: Beginning of stimulus time
    :type stimStart: int
    :param stimDuration: Duration of stimulus response
    :type stimDuration: int
    :returns: responseStats, list: average and standard deviation of the rate during the stimulus response
    """
    responseSpikeCount = []
    for k in spikeTrains.keys():
        spk = spikeTrains[k]

        spikes = spk.values
        spikes = np.array(spikes)
        # print spikes
        # print spikes[spikes < stimStart + stimDuration + 10]
        spikes = spikes[~np.isnan(spikes)]
        responseSpikeCount.append(len(spikes))

    if len(responseSpikeCount) > 0:
        responseStats = [np.mean(responseSpikeCount), np.std(responseSpikeCount)]
    else:
        responseStats = [0, 0]

    return responseStats