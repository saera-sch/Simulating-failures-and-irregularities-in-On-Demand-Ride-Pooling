def calculate_waiting_times_node_change_randomG(req_data,G):

    """
    Gets req_data from the simulation, where a failure occurred at 5000 requests and the graph the simulation was done on.
    Creates two arrays for every node in the graph saving all the wait times there before and after the failure and then averaging over these values.
    These arrays and the average values are saved into to seperate dictionaries, which are returned by the fuction.

    """
    waitnodes_before={}
    waitnodes_after={}
    waitn_avg_before={}
    waitn_avg_after={}

    for i in list(G.nodes):
        waitnodes_before[i]=[]
        waitnodes_after[i]=[]
        waitn_avg_before[i]=None
        waitn_avg_after[i]=None

    for key in req_data.keys():
        if key<5000:
            node=req_data[key]['origin']
            #if waitnodes[node]==None:
                #waitnodes[node]=(float(req_data[key]['pickup_epoch'])-float(req_data[key]['req_epoch']))
            #else:
            waitnodes_before[node].append((float(req_data[key]['pickup_epoch'])-float(req_data[key]['req_epoch'])))

        else:
            node=req_data[key]['origin']
            waitnodes_after[node].append((float(req_data[key]['pickup_epoch'])-float(req_data[key]['req_epoch'])))

    for node in waitnodes_before.keys():
        waitn_avg_before[node]=np.mean(waitnodes_before[node])
    for node in waitnodes_after.keys():
        waitn_avg_after[node]=np.mean(waitnodes_after[node])
    return waitn_avg_before, waitn_avg_after, waitnodes_before,waitnodes_after


def calculate_avg_waiting_times_node_change_cutedge_randomG(G,nG,x,l_avg,edge):

    """
    For one particular edge runs a simulation while removing the edge after 5000 requests.
    For every calculation the average wait time change at every single node is calculated with the function calculate_waiting_times_node_change_randomG(req_data,G).
    The values are saved into a dictionary. The mean value of node wait time of all simulations is calculated and saved into a new dictionary.
    the dictionary of all values before and after the calculation and the resulting dictionary with the average value for every node before and after the simulation is returned.
    """
    waitnodes_all_before={} #stores all the values of the before waiting time of all simulations
    waitnodes_all_after={}
    waitnodes_avg_before={} #stores the average of all simulations
    waitnodes_avg_after={}

    for i in list(G.nodes):
        waitnodes_all_before[i]=[]
        waitnodes_all_after[i]=[]
        waitnodes_avg_before[i]=None
        waitnodes_avg_after[i]=None

    for i in range(0,30):
        req_data1, insertion_data1,stoplist1,position1,time1,remaining_time1, next_stop1=simulate_single_request_rate(G, nG, x, network_type='novolcomp', l_avg=l_avg,initpos=None,time=0,remaining_time=0,next_stop=None,stoplist=[],req_data=dict(),insertion_data=[],req_idx=0)

        G.remove_edges_from(edge)
        nG = Network(G,network_type='novolcomp')
        #l_avg = nx.average_shortest_path_length(M)
        req_data, insertion_data,stoplist,position,time,remaining_time, next_stop=simulate_single_request_rate(G, nG, x, network_type='novolcomp', l_avg=l_avg,initpos=position1,time=time1,remaining_time=remaining_time1,next_stop=next_stop1,stoplist=stoplist1,req_data=req_data1,insertion_data=insertion_data1,req_idx=list(req_data1)[-1])

        waitn_avg_before, waitn_avg_after, waitnodes_before,waitnodes_after=calculate_waiting_times_node_change_randomG(req_data,G)
        for node in waitn_avg_before.keys():
            waitnodes_all_before[node].append(waitn_avg_before[node]) #goes through dictionary and appends the average value from this simulation to the array of overall values
        for node in waitn_avg_after.keys():
            waitnodes_all_after[node].append(waitn_avg_after[node])
        G.add_edges_from(edge)
        nG=Network(G,network_type='novolcomp')

    for node in waitnodes_all_before.keys():
        waitnodes_avg_before[node]=np.mean(waitnodes_all_before[node]) #saves the average of all simulation for every node into a new dictionary

    for node in waitnodes_all_after.keys():
        waitnodes_avg_after[node]=np.mean(waitnodes_all_after[node])


    return waitnodes_avg_before,waitnodes_avg_after,waitnodes_all_before,waitnodes_all_after
"""
The following shows how the data for the street network of Ingolstadt would be created and vizualized.
The same procedure was done for other networks.
"""

print("doing Ingolstadt")
graph_path = 'graph_ingolstadt.gpkl'
G = nx.read_gpickle(graph_path)
nG=Network(G,network_type='novolcomp')
l_avg = nx.average_shortest_path_length(G)
x=10
nx.edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None)
all_edges=G.edges()
local_edges={}
for edge in all_edges:
    print("doing Ingolstadt")
    graph_path = 'graph_ingolstadt.gpkl'
    G = nx.read_gpickle(graph_path)
    nG=Network(G,network_type='novolcomp')
    l_avg = nx.average_shortest_path_length(G)
    x=10
    edgelist=[]
    edgelist.append(edge)

    try:
        waitnodes_avg_before,waitnodes_avg_after,waitnodes_all_before,waitnodes_all_after=calculate_avg_waiting_times_node_change_cutedge_randomG(G,nG,x,l_avg,edgelist)
        #print(waitnodes_avg_before)
        #print(waitnodes_avg_after)
        before=[]
        after=[]
        #print(waitnodes_all_before)
        for i in waitnodes_avg_before.keys():
            before.append(waitnodes_avg_before[i])
        for i in waitnodes_avg_after.keys():
            after.append(waitnodes_avg_after[i])
        print(np.mean(before))
        print(np.mean(after))
        change={}
        for i in waitnodes_avg_before.keys():
            change[i]=waitnodes_avg_after[i]-waitnodes_avg_before[i]

        l_avg = nx.average_shortest_path_length(G)
        bet_centr=nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)

        G.remove_edges_from(edgelist)
        nG = Network(G,network_type='novolcomp')
        x=10
        l_avg = nx.average_shortest_path_length(G)
        #print(l_avg)
        bet_centr_after=nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
        G.add_edges_from(edgelist)
        nG = Network(G,network_type='novolcomp')
        change_betw_nodes={}
        for node in bet_centr.keys():
            change_betw_nodes[node]=bet_centr_after[node]-bet_centr[node]

        changes=[]
        for key in change.keys():
            changes.append(change[key])
        changes_betw=[]
        for key in change_betw_nodes.keys():
            changes_betw.append(change_betw_nodes[key])

        #print(changes)
        #print(changes_betw)
        local_edges[edge]={'changes':changes,'changes_betw':changes_betw}
    except KeyError:
        G.add_edges_from(edgelist)
        nG = Network(G,network_type='novolcomp')
        print(edge)
        continue

for edge in local_edges.keys():
    plt.scatter(local_edges[edge]['changes_betw'], local_edges[edge]['changes'],color='Blue')
plt.xlabel('Betweenness Centrality Change of Specific Node')
plt.ylabel('Change in Waiting Time of Specific Node')
