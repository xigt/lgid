
from collections import Counter


def get_window(mentions, top, bottom):
    for m in mentions:
        if m.startline <= bottom and m.endline >= top:
            yield m

def window_mention(feature, features, mentions, top, bottom):
    for m in get_window(mentions, top, bottom):
        features[(m.name, m.code)][feature] = True


def frequent_mention(feature, features, mentions, thresh, top, bottom):
    counts = Counter(
        (m.name, m.code) for m in get_window(mentions, top, bottom)
    )
    if thresh is None:
        thresh = max(counts.values())
    for pair, count in counts.items():
        if count >= thresh:
            features[pair][feature] = True


def closest_mention(feature, features, mentions, top, bottom, ref):
    window = sorted(
        (abs(ref - m.startline), m)
        for m in get_window(mentions, top, bottom)
    )
    if window:
        smallest_delta = window[0][0]
        for delta, mention in window:
            if delta > smallest_delta:
                break
            features[(mention.name, mention.code)][feature] = True


def in_line_mention(feature, features, mentions, line):
    for m in get_window(mentions, line.lineno, line.lineno):
        features[(m.name, m.code)][feature] = True


def gl_features(features, mentions, context, config):
    wsize = int(config['parameters']['window-size'])
    minfreq = int(config['parameters']['article-frequent-mention-threshold'])
    last = context['last-lineno']

    if config['features']['GL-first-lines']:
        window_mention('GL-first-lines', features, mentions, 0, wsize)
    
    if config['features']['GL-last-lines']:
        window_mention('GL-last-lines', features, mentions, last-wsize, last)
    
    if config['features']['GL-frequent']:
        frequent_mention('GL-frequent', features, mentions, minfreq, 0, last)
    
    if config['features']['GL-most-frequent']:
        frequent_mention('GL-most-frequent', features, mentions, None, 0, last)


def w_features(features, mentions, context, config):
    wsize = int(config['parameters']['window-size'])
    a_wsize = int(config['parameters']['after-window-size'])
    c_wsize = int(config['parameters']['close-window-size'])
    ac_wsize = int(config['parameters']['after-close-window-size'])
    minfreq = int(config['parameters']['frequent-mention-threshold'])
    a_minfreq = int(config['parameters']['after-frequent-mention-threshold'])

    t = context['span-top']
    b = context['span-bottom']

    if config['features']['W-prev']:
        window_mention('W-prev', features, mentions, t-wsize, t)

    if config['features']['W-close']:
        window_mention('W-close', features, mentions, t-c_wsize, t)

    if config['features']['W-closest']:
        closest_mention('W-closest', features, mentions, t-wsize, t, t)

    if config['features']['W-frequent']:
        frequent_mention('W-frequent', features, mentions, minfreq, t-wsize, t)

    if config['features']['W-after']:
        window_mention('W-after', features, mentions, b, b+a_wsize)

    if config['features']['W-close-after']:
        window_mention('W-close-after', features, mentions, b, b+ac_wsize)

    if config['features']['W-closest-after']:
        closest_mention('W-closest-after', features, mentions, b, b+a_wsize, b)

    if config['features']['W-frequent-after']:
        frequent_mention('W-frequent-after', features, mentions, minfreq, b,
                         b+a_wsize)


def l_features(features, mentions, olm, clm, context, config):
    if config['features']['L-in-line']:
        in_line_mention('L-in-line', features, mentions, line)
    if olm is not None:
        pass

    if clm is not None:
        pass

def g_features():
    pass

def m_features(features, mentions, line, config):
    if config['features']['M-in-line']:
        in_line_mention('M-in-line', features, mentions, line)
