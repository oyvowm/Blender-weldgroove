import OpenEXR as oe
import numpy as np
import Imath


def load_exr_asnp(file, layers=None, channels=None):
    file = oe.InputFile(file)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    output = []
    for lidx, layer in enumerate(layers):
        pt = file.header()['channels']['{:s}.{:s}'.format(layer, channels[lidx][0])].type
        if str(pt) == "HALF":
            dtype = np.float16
        elif str(pt) == "FLOAT":
            dtype = np.float32
        else:
            print("unsupported  type {:s} for layer {:s}".format(str(pt), layer))
            raise ValueError
        data = []
        for channel in channels[lidx]: 
            channel_str = file.channel('{:s}.{:s}'.format(layer, channel), pt)
            channel_num = np.frombuffer(channel_str, dtype = dtype)
            channel_num = channel_num.reshape(size[1], size[0])
            data.append(channel_num)
        
        output.append(np.stack(data, axis=2))

    if len(output) == 1:
        return output[0]
    else:
        return output

def load_exr_to_dict(file):
    '''
    Loads all layers into a dict where the keys are the layers.
    '''

    file = oe.InputFile(file)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    all_channels = file.header()['channels'].keys()
    layers = [k.split('.')[0] for k in file.header()['channels'].keys()]
    output_dict = dict.fromkeys(set(layers))


    for layer_name in output_dict.keys():
        output_dict[layer_name] = dict()

        layer_channels = []
        for channel in all_channels:
            if layer_name in channel: # E.g. 'rgb' in 'rgb.A'
                layer_channels.append(channel)

        for channel in layer_channels:
            
            dt = file.header()['channels'][channel].type
            if str(dt) == "HALF":
                dtype = np.float16
            elif str(dt) == "FLOAT":
                dtype = np.float32

            channel_str = file.channel(channel, dt)
            channel_np = np.frombuffer(channel_str, dtype = dtype)
            channel_np = channel_np.reshape(size[1], size[0])

            channel_suffix = channel.split('.')[-1]

            output_dict[layer_name][channel_suffix] = channel_np
            
    return output_dict



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize as Norm

    #rgb, d = load_exr_asnp('/home/grans/Documents/Blender/aluminium-corner-joint/Output/0019.exr',
    #            ['rgb', 'depth'], [['R', 'G', 'B'], ['V']])

    
    all_layers = load_exr_to_dict('/home/grans/Documents/Blender/aluminium-corner-joint/Output/0019.exr')

    rgb = plt.imread('/home/grans/Documents/Blender/aluminium-corner-joint/Output/rgb0019.png')

    rgb_exr = [all_layers['rgb'][el] for el in all_layers['rgb']]
    rgb_exr.reverse()
    rgb_exr = np.stack(rgb, axis=-1)
   
    depth = all_layers['depth']['V']

    lasermask = all_layers['lasermask']['V']

    pos = [all_layers['position'][el] for el in all_layers['position']]
    pos = np.stack(pos, axis=-1)

    fig, [rgb_ax, d_ax, mask_ax, pos_ax] = plt.subplots(1, 4, sharex=True, sharey=True)
    rgb_ax.imshow(rgb)
    d_ax.imshow(depth)
    mask_ax.imshow(lasermask)
    pos_ax.imshow(pos)
    
    plt.show()

    print("HOLD!")    
