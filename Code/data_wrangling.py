import numpy as np

def generate_odt_methoxy_data(training_fraction):
    methoxy = np.load('processed data\gold_odt_methoxy\processed_methoxy.npy')
    odt = np.load('processed data\gold_odt_methoxy\processed_odt.npy')
    np.random.shuffle(methoxy)
    np.random.shuffle(odt)
    print('initial shape of methoxy ', np.shape(methoxy))
    print('initial shape of odt ', np.shape(odt))
    print('max value of odt is ', np.max(odt))
    print('max value of methoxy is ', np.max(methoxy))
    # there are only half as many odt spectra as methoxy. This is a serious class imbalance.
    # taking same number of methoxy as odt.

    new_methoxy = methoxy[0:np.shape(odt)[0], :]
    print(' shape of new_methoxy ', np.shape(new_methoxy))

    # combining methoxy and odt data together and generating labels. methoxy=1 odt=0.
    print('creating labels odt = 0 and methoxy = 1 . Also combining data')
    methoxy_labels = np.ones(np.shape(new_methoxy)[0], dtype=int)
    odt_labels = np.zeros(np.shape(odt)[0], dtype=int)
    total_data = np.concatenate((new_methoxy, odt), 0)
    labels = np.concatenate((methoxy_labels, odt_labels), 0)

    print('shape of total data ', np.shape(total_data), 'shape of labels ', np.shape(labels))

    # randomly permuting methoxy and odt spectra
    print('randomly permuting data')
    p = np.random.permutation(len(total_data))
    total_data = total_data[p]
    labels = labels[p]

    print('element of lables', labels[0:50])

    #splitting data according to split fraction
    split_index = int(np.shape(total_data)[0] * training_fraction)

    training_spectra = total_data[:split_index]
    test_spectra = total_data[split_index:]
    training_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print('training spectra', np.shape(training_spectra), 'test spectra', np.shape(test_spectra))
    return [training_spectra, training_labels, test_spectra, test_labels]


def generate_odt_methoxy_gold_data(training_fraction):
    gold = np.load('processed data\gold_odt_methoxy\processed_gold.npy')
    methoxy = np.load('processed data\gold_odt_methoxy\processed_methoxy.npy')
    odt = np.load('processed data\gold_odt_methoxy\processed_odt.npy')
    np.random.shuffle(methoxy)
    np.random.shuffle(odt)
    np.random.shuffle(gold)
    print('initial shape of methoxy ', np.shape(methoxy))
    print('initial shape of odt ', np.shape(odt))
    print('initial shape of gold', np.shape(gold))
    print('max value of odt is ', np.max(odt))
    print('max value of methoxy is ', np.max(methoxy))
    print('max value of gold is ', np.max(gold))
    # there are only half as many odt spectra as methoxy. This is a serious class imbalance.
    # taking same number of methoxy as odt.

    new_methoxy = methoxy[0:np.shape(odt)[0], :]
    print(' shape of new_methoxy ', np.shape(new_methoxy))

    # combining methoxy and odt data together and generating labels. methoxy=[1, 0, 0] odt=[0, 1, 0] gold=[0,0,1].
    print('creating labels odt = [0, 1, 0], gold = [0, 0, 1] and methoxy = [1, 0, 0] . Also combining data')
    methoxy_labels = [int(0)]*np.shape(new_methoxy)[0]
    odt_labels = [int(1)]*np.shape(odt)[0]
    gold_labels = [int(2)]*np.shape(gold)[0]
    total_data = np.concatenate((new_methoxy, odt, gold), 0)
    labels = np.concatenate((methoxy_labels, odt_labels, gold_labels), 0)
    print('shape of total data ', np.shape(total_data), 'shape of labels ', np.shape(labels))

    # randomly permuting methoxy and odt spectra
    print('randomly permuting data')
    p = np.random.permutation(len(total_data))
    total_data = total_data[p]
    labels = labels[p]

    print('element of lables', labels[0:50])

    # splitting data according to split fraction
    split_index = int(np.shape(total_data)[0] * training_fraction)

    training_spectra = total_data[:split_index]
    test_spectra = total_data[split_index:]
    training_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print('training spectra', np.shape(training_spectra), 'test spectra', np.shape(test_spectra))
    return [training_spectra, training_labels, test_spectra, test_labels]


def generate_five_sams_data(training_fraction):
    odt = np.load('/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-solution/odt.npy')
    methoxy = np.load('/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-solution/meoht.npy')
    phhdt = np.load('/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-solution/phhdt.npy')
    h2f = np.load('/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-solution/h2f.npy')
    carborane = np.load('/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-solution/carborane.npy')
    #gold = np.load('processed data\multi sams data\processed_gold.npy')
    np.random.shuffle(odt)
    np.random.shuffle(methoxy)
    np.random.shuffle(phhdt)
    np.random.shuffle(h2f)
    np.random.shuffle(carborane)
    #np.random.shuffle(gold)

    print('initial shape of odt ', np.shape(odt))
    print('initial shape of methoxy ', np.shape(methoxy))
    print('initial shape of phhdt ', np.shape(phhdt))
    print('initial shape of h2f ', np.shape(h2f))
    print('initial shape of carborane ', np.shape(carborane))
    #print('initial shape of gold', np.shape(gold))


    print('max value of odt is ', np.max(odt))
    print('max value of methoxy is ', np.max(methoxy))
    print('max value of phhdt is ', np.max(phhdt))
    print('max value of h2f is ', np.max(h2f))
    print('max value of carborane is ', np.max(carborane))
    #print('max value of gold is ', np.max(gold))


    # combining sams data together and generating labels.
    #print('creating labels odt = [1, 0, 0, 0, 0], methoxy = [0, 1, 0, 0, 0], phhdt = [0, 0, 1, 0, 0]'
    #      ' h2f = [0, 0, 0, 1, 0], carborane = [0, 0, 0, 0, 1].'
    #     ' Also combining data')

    #odt_labels = [[1, 0, 0, 0, 0, 0]] * np.shape(odt)[0]
    #methoxy_labels = [[0, 1, 0, 0, 0, 0]] * np.shape(methoxy)[0]
    #phhdt_labels = [[0, 0, 1, 0, 0, 0]] * np.shape(phhdt)[0]
    #h2f_labels = [[0, 0, 0, 1, 0, 0]] * np.shape(h2f)[0]
    #carborane_labels = [[0, 0, 0, 0, 1, 0]] * np.shape(carborane)[0]
    #gold_labels = [[0, 0, 0, 0, 0, 1]]*np.shape(gold)[0]

    #odt_labels = [[1, 0, 0, 0, 0]] * np.shape(odt)[0]
    #methoxy_labels = [[0, 1, 0, 0, 0]] * np.shape(methoxy)[0]
    #phhdt_labels = [[0, 0, 1, 0, 0]] * np.shape(phhdt)[0]
    #h2f_labels = [[0, 0, 0, 1, 0]] * np.shape(h2f)[0]
    #carborane_labels = [[0, 0, 0, 0, 1]] * np.shape(carborane)[0]

    odt_labels = [int(0)] * np.shape(odt)[0]
    methoxy_labels = [int(1)] * np.shape(methoxy)[0]
    phhdt_labels = [int(2)] * np.shape(phhdt)[0]
    h2f_labels = [int(3)] * np.shape(h2f)[0]
    carborane_labels = [int(4)] * np.shape(carborane)[0]

    print("Label assignments: odt 0, methoxy 1, phhdt 2, h2f 3 and carborane 4.")
    #total_data = np.concatenate((odt, methoxy, phhdt, h2f, carborane, gold), 0)
    total_data = np.concatenate((odt, methoxy, phhdt, h2f, carborane), 0)
    #labels = np.concatenate((odt_labels, methoxy_labels, phhdt_labels, h2f_labels, carborane_labels, gold_labels), 0)
    labels = np.concatenate((odt_labels, methoxy_labels, phhdt_labels, h2f_labels, carborane_labels), 0)
    print('shape of total data ', np.shape(total_data), 'shape of labels ', np.shape(labels))

    # randomly permuting methoxy and odt spectra
    print('randomly permuting data')
    p = np.random.permutation(len(total_data))
    total_data = total_data[p]
    labels = labels[p]

    print('element of lables', labels[0:10])
    print('elements of data', total_data[0:10])

    # splitting data according to split fraction
    split_index = int(np.shape(total_data)[0] * training_fraction)

    training_spectra = total_data[:split_index]
    test_spectra = total_data[split_index:]
    training_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print('training spectra', np.shape(training_spectra), 'test spectra', np.shape(test_spectra))
    return [training_spectra, training_labels, test_spectra, test_labels]

def generate_five_sams_binned_data(training_fraction):
    #odt = np.load('processed data\odt.npy')
    #odtB = np.load('processed data\odtB.npy')
    #odtBB = np.load('processed data\odtBB.npy')
    odtBB = np.load('processed data\multi sams data-solution\Binned data\odt.npy')

    #methoxy = np.load('processed data\meoht.npy')
    #methoxyB = np.load('processed data\meohtB.npy')
    #methoxyBB = np.load('processed data\meohtBB.npy')
    methoxyBB = np.load('processed data\multi sams data-solution\Binned data\meoht.npy')

    #phhdt = np.load('processed data\phhdt.npy')
    #phhdtB = np.load('processed data\phhdtB.npy')
    #phhdtBB = np.load('processed data\phhdtBB.npy')
    phhdtBB = np.load('processed data\multi sams data-solution\Binned data\phhdt.npy')

    #h2f = np.load('processed data\h2f.npy')
    #h2fB = np.load('processed data\h2fB.npy')
    #h2fBB = np.load('processed data\h2fBB.npy')
    h2fBB = np.load('processed data\multi sams data-solution\Binned data\h2f.npy')

    #carborane = np.load('processed data\carborane.npy')
    #carboraneB = np.load('processed data\carboraneB.npy')
    #carboraneBB = np.load('processed data\carboraneBB.npy')
    carboraneBB = np.load('processed data\multi sams data-solution\Binned data\carborane.npy')


    #np.random.shuffle(odt)
    #np.random.shuffle(odtB)
    #np.random.shuffle(odtBB)
    np.random.shuffle(odtBB)

    #np.random.shuffle(methoxy)
    #np.random.shuffle(methoxyB)
    #np.random.shuffle(methoxyBB)
    np.random.shuffle(methoxyBB)

    #np.random.shuffle(phhdt)
    #np.random.shuffle(phhdtB)
    #np.random.shuffle(phhdtBB)
    np.random.shuffle(phhdtBB)

    #np.random.shuffle(h2f)
    #np.random.shuffle(h2fB)
    #np.random.shuffle(h2fBB)
    np.random.shuffle(h2fBB)

    #np.random.shuffle(carborane)
    #np.random.shuffle(carboraneB)
    #np.random.shuffle(carboraneBB)
    np.random.shuffle(carboraneBB)

    # combining sams data together and generating labels.

    #odt_labels = [int(0)] * (np.shape(odt)[0]+np.shape(odtB)[0]+np.shape(odtBB)[0]+np.shape(odtBBB)[0])
    #methoxy_labels = [int(1)] * (np.shape(methoxy)[0]+np.shape(methoxyB)[0]+np.shape(methoxyBB)[0]+np.shape(methoxyBBB)[0])
    #phhdt_labels = [int(2)] * (np.shape(phhdt)[0]+np.shape(phhdtB)[0]+np.shape(phhdtBB)[0]+np.shape(phhdtBBB)[0])
    #h2f_labels = [int(3)] * (np.shape(h2f)[0]+np.shape(h2fB)[0]+np.shape(h2fBB)[0]+np.shape(h2fBBB)[0])
    #carborane_labels = [int(4)] * (np.shape(carborane)[0]+np.shape(carboraneB)[0]+np.shape(carboraneBB)[0]+np.shape(carboraneBBB)[0])

    odt_labels = [int(0)] * (np.shape(odtBB)[0])
    methoxy_labels = [int(1)] * (np.shape(methoxyBB)[0])
    phhdt_labels = [int(2)] * (np.shape(phhdtBB)[0])
    h2f_labels = [int(3)] * (np.shape(h2fBB)[0])
    carborane_labels = [int(4)] * (np.shape(carboraneBB)[0])

    print("Label assignments: odt 0, methoxy 1, phhdt 2, h2f 3 and carborane 4.")

    #total_data = np.concatenate((odt, odtB, odtBB, odtBBB, methoxy, methoxyB, methoxyBB, methoxyBBB,
    #                             phhdt, phhdtB, phhdtBB, phhdtBBB, h2f, h2fB, h2fBB, h2fBBB,
    #                             carborane, carboraneB, carboraneBB, carboraneBBB), 0)

    total_data = np.concatenate((odtBB, methoxyBB,
                                 phhdtBB, h2fBB,
                                 carboraneBB), 0)

    labels = np.concatenate((odt_labels, methoxy_labels, phhdt_labels, h2f_labels, carborane_labels), 0)
    print('shape of total data ', np.shape(total_data), 'shape of labels ', np.shape(labels))

    # randomly permuting methoxy and odt spectra
    print('randomly permuting data')
    p = np.random.permutation(len(total_data))
    total_data = total_data[p]
    labels = labels[p]

    print('element of lables', labels[0:10])
    print('elements of data', total_data[0:10])

    # splitting data according to split fraction
    split_index = int(np.shape(total_data)[0] * training_fraction)

    training_spectra = total_data[:split_index]
    test_spectra = total_data[split_index:]
    training_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print('training spectra', np.shape(training_spectra), 'test spectra', np.shape(test_spectra))
    return [training_spectra, training_labels, test_spectra, test_labels]
