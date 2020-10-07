import numpy as np

def conv_t_size(inp, k, stride=1, out_pad=0, pad=0, dil=1):
    """Returns the output size of a ConvTranspose operation given the size of
    the square-shaped input `inp` and the kernel size `k`. Optional arguments
    are stride, `out_pad` (padding on one side of the output), `pad` (padding
    on both sides of the input), `dilation` (spacing between kernel elements).
    """
    inp = np.asarray(inp)
    out = (inp-1)*stride-2*pad+dil*(k-1)+out_pad+1
    if out.size > 1:
        return tuple(out)
    else:
        return out.item()


def conv_size(inp, k, stride=1, pad=0, dil=1):
    """Returns the output size of a Conv operation given the size of the
    square-shaped input `inp` and the kernel size `k`. Optional arguments are
    stride, `pad` (padding on both sides of the input), `dilation` (spacing
    between kernel elements).
    """
    inp = np.asarray(inp)
    out = (inp+2.*pad-dil*(k-1)-1)/stride+1
    if out.size > 1:
        return tuple(out)
    else:
        return out.item()
