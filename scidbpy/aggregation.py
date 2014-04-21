# License: Simplified BSD, 2014
# See LICENSE.txt for more information

import numpy as np

from . import SciDBArray
from .interface import _new_attribute_label
from .scidbarray import NP_SDB_TYPE_MAP

__all__ = ['histogram']


def histogram(X, bins=10, att=None, range=None, plot=False, **kwargs):
    """
    Build a 1D histogram from a SciDBArray.

    Parameters
    ----------
    X : SciDBArray
       The array to compute a histogram for
    att : str (optional)
       The attribute of the array to consider. Defaults to the first attribute.
    bins : int (optional)
       The number of bins
    range : [min, max] (optional)
       The lower and upper limits of the histogram. Defaults to data limits.
    plot : bool
       If True, plot the results with matplotlib
    histtype : 'bar' | 'step' (default='bar')
       If plotting, the kind of hisogram to draw. See matplotlib.hist
       for more details.
    kwargs : optional
       Additional keywords passed to matplotlib

    Returns
    -------
    (counts, edges [, artists])

        * edges is a NumPy array of edge locations (length=bins+1)
        * counts is the number of data betwen [edges[i], edges[i+1]] (length=bins)
        * artists is a list of the matplotlib artists created if *plot=True*
    """
    if not isinstance(X, SciDBArray):
        raise TypeError("Input must be a SciDBArray: %s" % type(X))
    if not isinstance(bins, int):
        raise NotImplementedError("Only integer bin arguments "
                                  "currently supported")
    f = X.afl
    binid = _new_attribute_label('bin', X)
    a = X.att(0) if att is None else att
    dtype = X.dtype if att is None else X.dtype[att]
    t = NP_SDB_TYPE_MAP[dtype.descr[0][1]]

    # store bounds
    if range is None:
        M = f.aggregate(X, 'min({a}) as min, max({a}) as max'.format(a=a))
        M = M.eval()
    else:
        lo = f.build('<min:%s NULL DEFAULT null>[i=0:0,1,0]' % t, min(range))
        hi = f.build('<max:%s NULL DEFAULT null>[j=0:0,1,0]' % t, max(range))
        M = f.cross_join(lo, hi, 'i', 'j').eval()

    val2bin = 'floor({bins} * ({a}-min)/(.0000001+max-min))'.format(bins=bins,
                                                                    a=a)
    bin2val = '{binid}*(0.0000001+max-min)/{bins} + min'.format(binid=binid,
                                                                bins=bins)

    schema = '<counts: uint64 null>[{0}=0:{1},1000000,0]'.format(binid, bins)
    s2 = ('<counts:uint64 null, min:{t} null, max:{t} null>'
          '[{binid}=0:{bins},1000000,0]'.format(binid=binid, t=t, bins=bins))

    # 0, min, max for each bin
    fill = f.slice(f.cross_join(f.build(schema, 0), M), 'i', 0).eval()
    fill2 = f.build('<v:int64>[i=0:0,1,0]', 0)  # single 0

    q = f.cross_join(X, M)  # val, min, max (Ndata)
    q = f.apply(q, binid, val2bin)  # val, min, max, binid
    q = f.substitute(q, fill2, binid)  # nulls to bin 0
    q = f.redimension(q, s2, 'count(%s) as counts' % binid)  # group bins
    q = f.merge(q, fill)  # replace nulls with 0
    q = f.apply(q, 'bins', bin2val)    # compute bin edges
    q = f.project(q, 'bins', 'counts')  # drop min, max

    result = q.toarray()

    assert result['counts'][-1] == 0
    ct, bin = result['counts'][:-1], result['bins']
    if plot:
        return ct, bin, _plot_hist(result, **kwargs)
    return ct, bin


def _plot_hist(result, **kwargs):
    import matplotlib.pyplot as plt
    histtype = kwargs.pop('histtype', 'bar')
    if histtype not in ['bar', 'step', 'stepfilled']:
        raise ValueError("histtype must be bar, step, or stepfilled")

    width = result['bins'][1] - result['bins'][0]

    if histtype == 'bar':
        x = result['bins'][:-1]
        y = result['counts'][:-1]
        return plt.bar(x, y, width, **kwargs)

    #histtype = step, stepfilled
    x = result['bins']
    x = np.hstack([[x[0]], x, [x[-1]]])
    y = np.hstack([0, result['counts'], 0])
    if histtype == 'stepfilled':
        x = np.column_stack([x[:-1], x[1:]]).ravel()
        y = np.column_stack((y[:-1], y[:-1])).ravel()
        return plt.fill(x, y, **kwargs)

    return plt.step(x, y, where='post', **kwargs)
