def parse_nodes(nodes_file):
    cell_dict = {}
    with open(nodes_file, 'r') as f:
        for line in f:
            line = line.strip() # 將這行字尾跟字首的空格與換行符號清掉
            if not line or line.startswith("UCLA") or line.startswith("#"): # 如果這一行是空的、或是開頭是 "UCLA"、或是註解（# 開頭），就跳過
                continue
            if line.startswith("NumNodes") or line.startswith("NumTerminals"):
                continue  # 跳過 NumNodes : xxx 等統計行
            parts = line.split() # 這是把這一行用空白切開，得到一個 list，例如 parts = ['a1', '2', '16']
            if len(parts) >= 3:
                name = parts[0]
                width = int(parts[1])
                height = int(parts[2])
                is_terminal = (len(parts) == 4 and 'terminal' in parts[3])
                cell_dict[name] = {
                    'w': width,
                    'h': height,
                    'terminal': is_terminal
                }
    return cell_dict

def parse_nets(nets_file):
    net_dict = {}
    current_net = None

    with open(nets_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("UCLA") or line.startswith("#"):
                continue

            if line.startswith("NumNets") or line.startswith("NumPins") :
                continue

            if line.startswith("NetDegree"):
                parts = line.split()
                net_name = parts[3] if len(parts) >= 4 else f"net_{len(net_dict)}"
                net_dict[net_name] = []
                current_net = net_name
            else:
                parts = line.split()
                if len(parts) >= 2:
                    cell = parts[0]
                    pin_type = parts[1]
                    x = y = None
                    if len(parts) >= 5:
                        try:
                            x = float(parts[3])
                            y = float(parts[4])
                        except:
                            pass
                    if current_net is not None:
                        net_dict[current_net].append({
                            'cell': cell,
                            'type': pin_type,
                            'x': x,
                            'y': y
                        })
                    else:
                        print(f"[Warning] Pin line found before any NetDegree line: {line}")

    return net_dict

def parse_pl(pl_file):
    pl_dict = {}
    with open(pl_file, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("UCLA") or line.startswith("#") :
                continue

            part = line.split()

            cell = part[0]
            x = float(part[1])
            y = float(part[2])
            orientation = part[4]
            fixed_ni = (len(part) == 6 and part[5] == '/FIXED_NI')
            # print("name = ", name, "x = ", x, "y = ", y, "orientation = ", orientation)
            pl_dict[cell] = {
                'x' : x,
                'y' : y,
                'orientation' : orientation,
                'fixed_ni' : fixed_ni
            }

    return pl_dict

class Scl:
    def __init__(self):
        self.coordinate = 0
        self.height = 0
        self.sitewidth = 0
        self.sitespacing = 0
        self.siteorient = ''
        self.sitesymmetry = ''
        self.subrowOrigin = ''
        self.numsites = 0
    def __str__(self):
        return (f"Coordinate: {self.coordinate}, Height: {self.height}, "
                f"SiteWidth: {self.sitewidth}, SiteSpacing: {self.sitespacing}, "
                f"SiteOrient: {self.siteorient}, SiteSymmetry: {self.sitesymmetry}, "
                f"SubrowOrigin: {self.subrowOrigin}, NumSites: {self.numsites}")
    # def __init__(self):
    #     pass

def parse_scl(scl_file):
    scl_list = []
    scl = None
    with open(scl_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith("UCLA") or line.startswith("#") :
                continue

            part = line.split()
            if part[0] == 'CoreRow':
                scl = Scl()
            elif part[0] ==   'Coordinate':
                scl.coordinate = part[2]
            elif part[0] ==   'Height':
                scl.height = part[2]
            elif part[0] ==   'Sitewidth':
                scl.sitewidth = part[2]
            elif part[0] ==   'Sitespacing':
                scl.sitespacing = part[2]
            elif part[0] ==   'Siteorient':
                scl.siteorient = part[2]
            elif part[0] ==   'Sitesymmetry':
                scl.sitesymmetry = part[2]
            elif part[0] ==   'SubrowOrigin':
                scl.subrowOrigin = part[2]
                scl.numsites = part[5]
                scl_list.append(scl)
    return(scl_list)


def set_net_pin(nets, pl):
    for net in nets:
        for net_index in nets[net]:
            # print(net_index)
            # print(net_index['x'])
            # print(pl[net_index['cell']]['x'])
            net_index['x'] = (0 if net_index['x'] is None else net_index['x']) + pl[net_index['cell']]['x']
            net_index['y'] = (0 if net_index['y'] is None else net_index['y']) + pl[net_index['cell']]['y']


if __name__ == '__main__':
    # file = "/home/b11107011/ic_cad/bookshelf/ICCAD04/ibm01/ibm01"
    file = "/home/b11107011/ic_cad/ICCAD25/aes_cipher_top/aes_cipher_top"
    cells = parse_nodes(file + ".nodes") #確定OK
    nets = parse_nets(file + ".nets") #確定OK
    # print(list(nets.keys()))
    # print(nets["key[127]"])
    pl = parse_pl(file + ".pl") #確定OK
    print(pl['text_out[0]'])
    print(pl['FE_DBTC121_ld'])
    scl = parse_scl(file + ".scl") #確定OK
    # for s in scl:
    #     print(s)
    set_net_pin(nets, pl)
    # print(nets["key[127]"])
    # print(cells)
    # print(nets["net0"])