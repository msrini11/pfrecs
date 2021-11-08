# Explainability of random walk
# Author: Srinivasan M, PathFactory
# The aim of Explainability class is two fold - Narrative and Visual - of the Random Walk Algo
# Works off subgraph in the form of nodes and links output resulting from a rw approach
import networkx as nx
import math
import statistics
from collections import OrderedDict


class Explainability:
    node_type_config = {
        "product": ("green", 2),
        "document": ("pink", 3),
        "country": ("azure", 5),
        "industry": ("brown", 7),
        "topic": ("grey", 11),
        "region": ("lightblue", 13),
        "keywords": ("lightgrey", 17),
        "topic-word": ("orange", 19)
    }
    # Node type color map - to color the node type with these colors
    # red is not in here as it is later used as the color of the result url node
    # any other color not mentioned here defaults to black. So black, red not to be in the above
    node_type_c = {x: y[0] for x, y in node_type_config.items()}
    # Node type number ids. They are primes
    node_type_n = {x: y[1] for x, y in node_type_config.items()}
    # separator for node type and node name to uniquely identify them
    attrib_sep = "--#--"
    # node shapes are circular but given start nodes have square shape
    def __init__(self, thresh, node_opt="name", filter_opt=0):
        # Node option can take: 'id' or 'name'
        self.node_option = node_opt
        # threshold specifies the min weight. It can take a float number or string "avg"
        # avg involves finding the mean of weights and taking it as a threshold
        self.weight_thresh = thresh
        self.start_points = []
        self.result_points = []
        self.node_types = [x for x,y in self.node_type_config.items() if filter_opt % y[1] == 0] if filter_opt > 0 else []

    def set_thresh(self, thresh):
        # Threshold for filtering out lighter weights
        self.weight_thresh = thresh

    def controller(self, dat, start_points=None, consolidate=0, layout="c"):
        result_url = dat.get("normalized_url")     # dat.get("url")
        self.result_points = self.make_result_points(result_url)
        self.start_points = start_points if start_points else {}
        # Topic option indicator is to extract topic words from the topic, if present
        topic_option = 1 if self.node_types and "topic-word" in self.node_types else 0
        # Prepare the given data element (dat)
        nodes, links = self.prep(dat, topic_option)
        # Node types defined in the config for selective filter. It is a soft filter.
        if self.node_types:
            print(f"Node Types considered: {self.node_types}")
            nodes, links = self.filter_with(self.node_types, nodes, links)
        aggr_info = {}
        # Consolidation indicator
        if consolidate > 0:
            nts = list(self.node_type_c.keys())
            if consolidate == 1:
                nodes, links = self.consolidate_edges_from_nodetypes(nodes, links, nts, nts)
            elif consolidate == 2:
                # Consolidating node types from start nodes on the subgraph
                self.typename_id_map = {self.pack_nodetype_name(x.get("node_type"), x.get("name")): x["id"] for x in nodes}
                self.id_name_map = {x["id"]: x.get("name") for x in nodes}
                result_ids = [self.typename_id_map.get(x) for x in self.result_points]
                grph = self.grapher(nodes, links)
                nodes, links, aggr_info = self.aggregate_graph(grph, None, result_ids)
            elif consolidate == 3:
                # for labeling there is a need to consolidate based on implicit and explicit start nodes
                self.typename_id_map = {self.pack_nodetype_name(x.get("node_type"), x.get("name")): x["id"] for x in nodes}
                self.id_name_map = {x["id"]: x.get("name") for x in nodes}
                grph = self.grapher(nodes, links)
                # Labeler returns a new set of nodes and links
                # They need not be used, so getting them in other variables
                # output of interest is aggr_info
                nodes1, links1, aggr_info = self.labeler(grph)
        nodes = self.tfm(nodes, self.node_option)
        self.typename_id_map = {self.pack_nodetype_name(x.get("node_type"), x.get("name")): x["id"] for x in nodes}
        self.id_name_map = {x["id"]: x.get("name") for x in nodes}
        grph = self.grapher(nodes, links)
        # visualise the graph
        self.viz_graph(grph, nodes, consolidate, layout)
        return grph, nodes, links, aggr_info

    def nodes_info(self, nodes):
        info = {}
        for nodetype in self.node_type_config:
            info[nodetype] = [x["name"] for x in nodes if x["node_type"] == nodetype]
        return info

    def tmp_remove_node(self, node_type, node_name, nodes, links):
        typename = self.pack_nodetype_name(node_type, node_name)
        typename_id_map = {self.pack_nodetype_name(x.get("node_type"), x.get("name")): x["id"] for x in nodes}
        n = typename_id_map.get(typename)
        nodes = [x for x in nodes if x["id"] != n]
        links = [x for x in links if x["source"] !=n and x["target"] != n]
        return nodes, links

    def labeler(self, grph):
        # Used to come up with textual labels based on the graph
        # Result id is 1 or more result URLs
        # bc captures info flow thru a node. Bridge - how critical a node despite lower weights
        # bc = nx.betweenness_centrality(grph, weight='weight')
        # dc = nx.degree_centrality(grph)
        # ec = nx.eigenvector_centrality(grph, max_iter=1000, weight='weight')
        # start_ids = [self.typename_id_map.get(x) for x in self.start_points]
        # print ("START",start_ids)
        # for node in grph.nodes:
        #     x = bc[node]
        #     y = dc[node]
        #     # measure of the influence a node has on a network
        #     z = ec[node]
        #     print(f"{node}-{self.shorten_name(self.id_name_map[node]).rjust(30)}: {str(round(x,2)).ljust(5)} {str(round(y,2)).ljust(5)} {round(z,2)}")
        # deg1 = grph.in_degree(weight='weight')
        # deg1 = sorted(deg1, key=lambda x: x[1], reverse=True)
        # print(deg1)
        result_ids = [self.typename_id_map.get(x) for x in self.result_points]
        deg2 = grph.out_degree(weight='weight')
        deg2 = sorted(deg2, key=lambda x: x[1], reverse=True)
        print("DEG2", deg2)
        print("Result IDS", result_ids)
        # from_ids = [x[0] for x in deg2 if x[0] not in result_ids]
        # p = nx.algorithms.descendants(grph, result_ids[0])
        # Aggregate based on explicit (given) and implicit (indegree 0) start nodes. None implies this
        nodes, links, aggr_info = self.aggregate_graph(grph, None, result_ids)
        return nodes, links, aggr_info

    def get_results(self, grph, links, aggr_info):
        # Assemble the labeler results
        result_ids = [self.typename_id_map.get(x) for x in self.result_points]
        start_ids = [self.typename_id_map.get(x) for x in self.start_points]
        results = {}
        # low/high weight levels are calc. Grey area for now, but works.
        # SD does not yield well as it is large, given the huge variability in weights
        lwl = self.avg_weight - (0.6 * self.avg_weight)
        hwl = self.avg_weight + (0.2 * self.avg_weight)
        print (f"LOW UPPER LIMITS {lwl}, {hwl} AVG={self.avg_weight} SD={self.std_dev}")
        for i in result_ids:
            lnks = [x for x in links if x["target"] == i]
            lnks.sort(key=lambda x: x["weight"], reverse=True)
            factors = []
            for lnk in lnks:
                name = self.id_name_map.get(lnk['source'])
                weight = lnk.get('weight', 0)
                giv = 'given' if lnk['source'] in start_ids else ''
                nt = lnk['node_type_s']
                influence_level = "medium" if lwl < weight < hwl else 'strong' if weight >= hwl else 'light'
                if nt == name and name in aggr_info:
                    # See aggr_info for more info
                    add_inf = OrderedDict(sorted(aggr_info[name].items(), key=lambda x:x[1], reverse=True))
                    addinf = [x for x in add_inf]
                    ellipsis = "..." if len(addinf)>3 else ""
                    name = "(" + ", ".join(addinf[:3]) + ellipsis + ")"
                factors.append({
                    'il': influence_level,
                    'nm': name,
                    'giv': giv,
                    'nt': nt
                })
            results.update({self.id_name_map.get(i): factors})
            # Now process indirect influences - those that contribute to the result node(s)
            # Trace the path of outgoing nodes from these, resulting into the result node(s)
            # Select among strong/medium/light as deeper as required
            deeper_levels = ['strong']
            for idx in range(len(lnks)):
                lnk = lnks[idx]
                if factors[idx]['il'] not in deeper_levels:
                    continue
                src = lnk['source']
                trg = lnk['target']
                thru_src = self.id_name_map.get(src)
                weight = 1
                if not grph.has_edge(src, trg):
                    continue
                shortest_paths = nx.all_shortest_paths(grph, src, trg, weight='weight')
                for paths in map(nx.utils.pairwise, shortest_paths):
                    thru_names = []
                    for path in paths:
                        if path == (src, trg):
                            continue
                        weight *= grph[path[0]][path[1]]['weight'] / self.avg_weight if self.avg_weight > 0 else 1
                        thru_names.append(self.shorten_name(self.id_name_map.get(path[1])))
                    if thru_names:
                        influence_level = "medium" if lwl < weight < hwl else 'strong' if weight >= hwl else 'light'
                        ellipsis = "..." if len(thru_names) > 3 else ""
                        name = "(" + ", ".join(thru_names[:3]) + ellipsis + f" via {thru_src}" + ")"
                        factors.append({
                            'il': influence_level,
                            'nm': name,
                            'giv': "",
                            'nt': "nodes"
                        })
        return results

    def apply_markdown(self, results):
        # Apply markdown to results. Tweak this part as necessary
        sr = 0  # Serial number of rec doc
        lines = []
        for doc in results:
            sr += 1
            factors = results[doc]
            strs = []
            # Remove duplicate entries (if nt and nm are the same)
            prevs = []
            # All previous entries are stored in prevs
            # factors already sorted (strong appears first since weight is more and light appears later if any)
            for idx in range(len(factors)):
                x = factors[idx]
                curr = f"{x.get('nt', '')}-{x.get('nm', '')}"
                if curr in prevs:
                    continue
                else:
                    prevs.append(curr)
                    if x['nm']:
                        strs.append(f"{x['il']} influence from {x['giv']} *{x['nt']}* `{x['nm']}`")
            strs = [x.replace("strong influence", "**strong** influence") for x in strs]
            strs = [x.replace("given", "(given)") for x in strs]
            strs = [x.replace(" via ", "` via `") for x in strs]

            lines.append(f"{sr}. Recommends *{doc}* because of:\n")
            for x in strs:
                lines.append(f"+ {x}")  # + is bulleted
        return lines

    def apply_markdown2(self, results):
        # Apply markdown to results. Tweak this part as necessary
        sr = 0      # Serial number of rec doc
        docline_obj = OrderedDict()
        for doc in results:
            sr += 1
            factors = results[doc]
            strs = []
            # Remove duplicate entries (if nt and nm are the same)
            prevs = []
            # All previous entries are stored in prevs
            # factors already sorted (strong appears first since weight is more and light appears later if any)
            for idx in range(len(factors)):
                x = factors[idx]
                curr = f"{x.get('nt', '')}-{x.get('nm', '')}"
                if curr in prevs:
                    continue
                else:
                    prevs.append(curr)
                    if x['nm']:
                        strs.append(f"{x['il']} influence from {x['giv']} *{x['nt']}* `{x['nm']}`")
            strs = [x.replace("strong influence", "**strong** influence") for x in strs]
            strs = [x.replace("given", "(given)") for x in strs]
            strs = [x.replace(" via ", "` via `") for x in strs]

            # lines.append(f"{sr}. Recommends *{doc}* because of:\n")
            doclines = []
            for x in strs:
                doclines.append(f"+ {x}") # + is bulleted
            docline_obj.update({f"{sr}. {doc}": doclines})
        return docline_obj

    def make_result_points(self, result_urls):
        if type(result_urls) is not list:
            result_urls = [result_urls]
        result_points = [self.pack_nodetype_name("document", x) for x in result_urls]
        return result_points

    def grapher(self, nodes, links):
        grph = nx.DiGraph()
        grph_nodes = [(x["id"], {"node_type": x["node_type"], "name":x["name"]}) for x in nodes]
        grph.add_nodes_from(grph_nodes)
        print (f"-------THRESH GRAPHER = {self.weight_thresh}")
        edges = self.get_edges(links, self.weight_thresh)
        grph.add_weighted_edges_from(edges)
        return grph

    def combine_data(self, data):
        # If all results are selected, the nodes will have to be intelligently combined
        # Below algorithm needs improvement, though. How to remove duplicates?
        doc_name_set = set()
        result_urls = []
        c_data = {}
        for dat in data:
            result_urls.append(dat.get("normalized_url"))      # "url"
            nodes, links = self.prep(dat)
            # update the dat to include the combined nodes, links
            dat["nodes"] = nodes
            dat["links"] = links
            doc_name_set = doc_name_set.union({x["name"] for x in nodes})
        doc_names = list(doc_name_set)
        doc_id_map = {doc_names[x]:x for x in range(len(doc_names))}
        c_nodes = []
        c_links = []
        for dat in data:
            xpl = dat.get("explainability", {})
            nodes = xpl.get("nodes", [])
            links = xpl.get("links", [])
            for link in links:
                docname_s = next(filter(lambda x:x["id"]==link["source"], nodes), {})
                src = doc_id_map.get(docname_s.get("name"))
                docname_t = next(filter(lambda x: x["id"] == link["target"], nodes), {})
                trg = doc_id_map.get(docname_t.get("name"))
                # Check for any duplicate in edges. The weights are to be added instead of adding a new edge
                duplink = next(filter(lambda x: src==x["source"] and trg==x['target'], links), None)
                if duplink:
                    # link['source'] = -999
                #    if 'weight' in duplink:
                #        duplink['weight'] += 0  # link.get('weight', 1)
                    continue
                link["source"] = doc_id_map.get(docname_s.get("name"))
                link["target"] = doc_id_map.get(docname_t.get("name"))

            for node in nodes:
                node["id"] =  doc_id_map.get(node["name"])
            c_nodes.extend(nodes)
            c_links.extend(links)
        # ---------------
        # c_links = [x for x in c_links if x['source'] != -999]
        # c_data.update({"explainability": {"nodes": c_nodes, "links": c_links}, "url": result_urls})
        c_data.update({"explainability": {"nodes": c_nodes, "links": c_links}, "normalized_url": result_urls})
        return c_data

    def prep(self, dat, topic_option=0):
        xpl = dat.get("explainability", {})
        nodes = xpl.get("nodes", [])
        links = xpl.get("links", [])
        id_nodetype_map = {x.get("id"): x.get("node_type") for x in nodes}
        self.add_node_types_2links(id_nodetype_map, links)
        # Store the average before filtering
        self.min_weight, self.avg_weight, self.max_weight, self.std_dev = self.get_avg_weight(links)
        # print (f"Weights Min-{self.min_weight}, Avg-{self.avg_weight}, Max-{self.max_weight}")
        if self.weight_thresh == "avg":
            nodes, links = self.remove_lighter_avg(nodes, links)
        else:
            nodes, links = self.remove_lighter(nodes, links, self.weight_thresh)
        if topic_option > 0:
            nodes,links = self.connect_topic_keywords(nodes, links)
        return nodes, links

    def add_node_types_2links(self, id_nodetype_map, links):
        for x in links:
            x.update({
                "node_type_s": id_nodetype_map.get(x["source"]),
                "node_type_t": id_nodetype_map.get(x["target"])
            })
        return

    def connect_topic_keywords(self, nodes:list, links:list):
        topic_nodes = [x for x in nodes if x["node_type"] == "topic"]
        def get_topic_node(name):
            tw_name_id_map = {x.get("name"):x.get("id") for x in nodes if "topic-word" == x.get("node_type")}
            if name in tw_name_id_map:
                id = tw_name_id_map[name]
                return True, id
            try:
                id = 1 + max([x.get("id") for x in nodes])
            except Exception as e:
                id = len(nodes)
            return False, id
        for node in topic_nodes:
            top_words = [x for x in node.get("top_words", [])]
            source_id = node["id"]
            source_node_type = node["node_type"]
            # normalize the weights
            sumw = sum(t.get("prob", 0) for t in top_words)
            # add node: node_type=topic-keyword, name=<word>, id=num
            for tw in top_words:
                word = tw.get("word")
                exists, target_id = get_topic_node(word)
                if not exists:
                    # print (f"Appending Node: {word}-{target_id} to topic-word")
                    nodes.append({
                        "name": word,
                        "id": target_id,
                        "node_type": "topic-word"
                    })
                # weight = tw.get("prob",0) / sumw if sumw > 0 else 0
                weight = 10 * tw.get("prob", 0)
                # print (f"Link - {source_id}->{target_id}, {source_node_type} to topic-word")
                links.append({
                    "node_type_s": source_node_type,
                    "node_type_t": "topic-word",
                    "weight": weight,
                    "source": source_id,
                    "target": target_id
                })
        return nodes,links

    def get_edges(self, links, thresh=None):
        if thresh:
            edges = [(x["source"], x["target"], x["weight"]) for x in links if x["weight"] > thresh]
        else:
            edges = [(x["source"], x["target"], x["weight"]) for x in links]
        return edges

    def filter_with(self, node_types, nodes, links):
        node_ids = {x["id"] for x in nodes if x["node_type"] in node_types}
        links = [x for x in links if x["source"] in node_ids or x["target"] in node_ids]
        #
        participating_node_ids = {x["source"] for x in links}.union({x["target"] for x in links})
        nodes = [x for x in nodes if x["id"] in participating_node_ids]
        nodes, links = self.remove_lighter(nodes, links, self.weight_thresh)
        return nodes, links

    def tfm(self, nodes, label_ind):
        # label indicator: id or name
        # Transform to suit name rather than id. Better to display
        # id_url_map = {x.get("id"): self.shorten_name(x.get("name")) for x in nodes}
        for node in nodes:
            node.update({
                "label": self.shorten_name(node.get("name")) if label_ind == "name" else node.get("id")
            })
        return nodes

    # def remove_lighter(self, nodes:list, links:list, thresh:float)->tuple[list, list]:
    def remove_lighter(self, nodes: list, links: list, thresh: float):
        # edges = self.get_edges(links, thresh)
        participating_node_ids = {x["source"] for x in links if x["weight"] > thresh}.union({x["target"] for x in links if x["weight"] > thresh})
        # participating_node_ids = {x[0] for x in edges}.union(x[1] for x in edges)
        links = [x for x in links if x["source"] in participating_node_ids or x["target"] in participating_node_ids]
        nodes = [x for x in nodes if x["id"] in participating_node_ids]
        return nodes, links

    def remove_lighter_avg(self, nodes, links):
        minw, avgw, maxw, sd = self.get_avg_weight(links)
        return self.remove_lighter(nodes, links, avgw)

    def get_avg_weight(self, links):
        weights = [x.get("weight", 1) for x in links]
        avgw = statistics.mean(weights) if len(weights) > 0 else 0.0
        minw = min(weights) if len(weights) > 0 else 0
        maxw = max(weights) if len(weights) > 0 else 0
        std_dev = statistics.stdev(weights) if len(weights) > 1 else 0.0
        return 1.0*minw, 1.0*avgw, 1.0*maxw, std_dev

    def consolidate_edges_from_nodetypes(self, nodes, links, from_node_types, to_node_types):
        if len(links) == 0:
            return []
        c_links = []
        c_nodes = []
        def get_node(name):
            names = [x.get("name") for x in c_nodes]
            if name in names:
                idx = names.index(name)
                return True, idx
            try:
                idx = 1+max([x.get("id") for x in c_nodes])
            except Exception as e:
                idx = len(c_nodes)
            return False, idx
        id_name_map = {x.get("id"): x.get("name") for x in nodes}
        id_nodetype_map = {x.get("id"): x.get("node_type") for x in nodes}
        nlinks_accounted = 0
        # Assume the source and target types are added already
        for from_node_type in from_node_types:
            for to_node_type in to_node_types:
                links_x = [x for x in links if x["node_type_t"] == to_node_type and x["node_type_s"] == from_node_type]
                if len(links_x) == 0:
                    continue
                nlinks_accounted += len(links_x)
                weight = sum([w["weight"] for w in links_x])
                # Create nodes based on source and target
                exists, node_id_s = get_node(from_node_type)
                if not exists:
                    c_nodes.append({
                        "name": from_node_type,
                        "id": node_id_s,
                        "node_type": from_node_type
                    })
                exists, node_id_t = get_node(to_node_type)
                if not exists:
                    c_nodes.append({
                        "name": to_node_type,
                        "id": node_id_t,
                        "node_type": to_node_type
                    })
                c_links.append({
                    "node_type_s": from_node_type,
                    "node_type_t": to_node_type,
                    "weight": weight,
                    "source": node_id_s,
                    "target": node_id_t
                })
        # print(f"Links: {len(links)}, acctd: {nlinks_accounted}, c_nodes: {len(c_nodes)}, c_links: {len(c_links)}")
        # print ([f"{x['name']}, {x['node_type']}, {x['id']}" for x in c_nodes])
        # print ([f"{x['source']}->{x['target']} {x['weight']}" for x in c_links])
        return c_nodes, c_links

    def unpack_node_point(self, node_point: str):
        # Node Point is a group of node type and name separated by a unique sep
        if self.attrib_sep not in node_point:
            return ("", "")
        parts = node_point.partition(self.attrib_sep)
        node_type = parts[0]
        name = parts[1] if len(parts) > 1 else None
        return (node_type, name)

    def pack_nodetype_name(self, node_type, name):
        return self.attrib_sep.join([node_type, name])

    def aggregate_graph(self, grph, from_ids=None, to_ids=None):
        # From and to ids are list of node ids
        # Roots are node ids where the node does not have an incoming link. Plus add the starting node ids
        start_ids = list({self.typename_id_map.get(p) for p in self.start_points if self.typename_id_map.get(p) is not None})
        if from_ids is None:
            # Assume all start points
            roots = list({
                self.typename_id_map.get(p) for p in self.start_points if self.typename_id_map.get(p) is not None
            }.union({
                v for v,d in grph.in_degree() if d==0
            }))
        else:
            roots = from_ids
        # Leaves are where the edges end
        if to_ids is None:
            leaves = [v for v, d in grph.out_degree() if d == 0]
        else:
            leaves = to_ids
        all_paths = []
        for root in roots:
            for leaf in leaves:
                try:
                    if grph.has_edge(root, leaf):
                        # paths = nx.shortest_simple_paths(grph, root, leaf, weight='weight')
                        paths = nx.algorithms.simple_paths.all_simple_paths(grph, root, leaf)
                        for path in map(nx.utils.pairwise, paths):
                            # print (nx.path_weight(grph, path, 'weight'))
                            all_paths.extend(path)
                except Exception as e:
                    continue

        id_type_map = nx.get_node_attributes(grph, "node_type")
        id_name_map = nx.get_node_attributes(grph, "name")
        nodetype_paths = [
            (
                id_type_map.get(x[0]) if x[0] not in roots+leaves else x[0],
                id_type_map.get(x[1]) if x[1] not in roots+leaves else x[1]
            ) for x in all_paths
        ]
        weights = [grph[x[0]][x[1]]["weight"] for x in all_paths]
        new_edges = []
        new_weights = []
        aggrd_info = {}
        def store_aggr(edg, path, wt):
            for i in [0,1]:
                if edg[i] != path[i]:
                    if edg[i] not in aggrd_info:
                        aggrd_info.update({edg[i]:{}})
                    if path[i] not in aggrd_info[edg[i]]:
                        aggrd_info[edg[i]].update({path[i]: 0})
                    aggrd_info[edg[i]][path[i]] += wt
        for idx in range(len(nodetype_paths)):
            edg = nodetype_paths[idx]
            store_aggr(edg, all_paths[idx], weights[idx])
            if edg not in new_edges:
                new_edges.append(edg)
                new_weights.append(weights[idx])
            else:
                newidx = new_edges.index(edg)
                new_weights[newidx] = weights[idx]      ###### +=
        maxid = 0 if len(id_type_map) == 0 else max(id_type_map)
        # See if it is already there otherwise they need to be created
        new_node_ids = {x[0] for x in new_edges}.union({x[1] for x in new_edges})
        tbc_nodes_names = [y for y in new_node_ids if y not in id_type_map]
        tbc_ids = [x+1+maxid for x in range(len(tbc_nodes_names))]

        new_nodes = [
            {
                "id": idval,
                "node_type":id_type_map.get(idval),
                "name": id_name_map.get(idval)
            } for idval in new_node_ids if idval not in tbc_nodes_names
        ] + [
            {
                "id": tbc_ids[idx],
                "node_type": tbc_nodes_names[idx],
                "name": tbc_nodes_names[idx]
            } for idx in range(len(tbc_ids))
        ]
        new_idtype_map = {x['id']: x['node_type'] for x in new_nodes}
        new_idname_map = {x['id']: x['name'] for x in new_nodes}
        new_edges = [
            (
                tbc_ids[tbc_nodes_names.index(e[0])] if e[0] in tbc_nodes_names else e[0],
                tbc_ids[tbc_nodes_names.index(e[1])] if e[1] in tbc_nodes_names else e[1]
            ) for e in new_edges
        ]
        new_links = [
            {
                "source": new_edges[idx][0],
                "target": new_edges[idx][1],
                "weight": new_weights[idx],
                "node_type_s": new_idtype_map.get(new_edges[idx][0]),
                "node_type_t": new_idtype_map.get(new_edges[idx][1])
            } for idx in range(len(new_edges))
        ]
        #
        for d in aggrd_info:
            aggrd_info[d] = {id_name_map.get(x) if id_name_map.get(x) is not None else new_idname_map.get(x):aggrd_info[d][x] for x in aggrd_info[d]}
        # print ("AGGRDINF=", aggrd_info)
        return new_nodes, new_links, aggrd_info

    def consolidate_from_nodes(self, nodes, links, from_nodes, to_node_types):
        id_url_map = {x.get("id"): x.get("name") for x in nodes}
        id_nodetype_map = {x.get("id"): x.get("node_type") for x in nodes}
        c_links = []
        c_nodes = []
        def get_node(name):
            names = [x.get("name") for x in c_nodes]
            if name in names:
                idx = names.index(name)
                return True, idx
            try:
                idx = 1+max([x.get("id") for x in c_nodes])
            except Exception as e:
                idx = len(c_nodes)
            return False, idx
        nlinks_accounted = 0
        for from_node in from_nodes:
            for to_node_type in to_node_types:
                links_x = [x for x in links if x["node_type_t"] == to_node_type and x["source"] == from_node]
                if len(links_x) == 0:
                    continue
                nlinks_accounted += len(links_x)
                weight = sum([w["weight"] for w in links_x])
                # Create nodes based on source and target
                exists, node_id_s = get_node(from_node)
                c_nodes.append({
                    "name": from_node,
                    "id": len(c_nodes),
                    "node_type": id_nodetype_map.get(from_node, from_node)
                })
                c_links.append({
                    "node_type_s": id_nodetype_map.get(from_node, from_node),
                    "node_type_t": to_node_type,
                    "weight": weight,
                    "source": from_node,
                    "target": to_node_type
                })
                if from_node not in [x.get("name") for x in c_nodes]:
                    c_nodes.append({
                        "name": from_node,
                        "id": len(c_nodes),
                        "node_type": id_nodetype_map.get(from_node, from_node)
                    })
                if to_node_type not in [x.get("name") for x in c_nodes]:
                    c_nodes.append({
                        "name": to_node_type,
                        "id": len(c_nodes),
                        "node_type": to_node_type
                    })
        # participating_node_ids = {x["source"] for x in c_links}.union({x["target"] for x in c_links})
        # c_nodes = [{"name": x, "node_type": id_nodetype_map.get(x, x), "id": x} for x in participating_node_ids]
        # print(f"LinksSN: {len(links)}, acctd: {nlinks_accounted}, c_nodes: {len(c_nodes)}, c_links: {len(c_links)}")
        # print ([x["weight"] for x in c_links])
        return c_nodes, c_links

    def viz_graph(self, grph, nodes, consolidate=0, layout_opt="c"):
        edgs = grph.edges()
        if layout_opt == "c":
            pos = nx.layout.circular_layout(grph)
        elif layout_opt == "f":
            pos = nx.layout.fruchterman_reingold_layout(grph)
        elif layout_opt == "s":
            pos = nx.layout.spring_layout(grph)
        elif layout_opt == "p":
            pos = nx.layout.planar_layout(grph) if nx.check_planarity(grph)[0] else nx.layout.spring_layout(grph)
        else:
            return
        typename_id_map = {self.pack_nodetype_name(x.get("node_type"), x.get("name")): x["id"] for x in nodes}

        # startpoints: node_type<sep><node_name. e.g. document--#--eaton
        starts = [x for x in typename_id_map if x in self.start_points]
        color_starts = ["red" if x in self.result_points else self.node_type_c.get(self.unpack_node_point(x)[0], "black") for x in starts]
        start_ids = [typename_id_map[x] for x in starts]
        non_starts = [x for x in typename_id_map if x not in self.start_points]
        non_start_ids = [typename_id_map[x] for x in non_starts]
        color_non_starts = ["red" if x in self.result_points else self.node_type_c.get(self.unpack_node_point(x)[0], "black") for x in non_starts]

        weights = self.normalize_weights([grph[u][v]['weight'] for u, v in edgs])
        nx.draw_networkx_nodes(
            grph, pos, alpha=0.9,
            nodelist=non_start_ids,
            node_shape="o",
            node_color=color_non_starts
        )

        nx.draw_networkx_nodes(
            grph, pos, alpha=0.81,
            nodelist=start_ids,
            node_color=color_starts,
            node_shape="s"
        )
        if consolidate == 0:
            nx.draw_networkx_edges(grph, pos, width=weights, alpha=0.5, arrows=True)
        else:
            # Bring in a slight twist in the arrow edge to vividly display bi-directional edges
            nx.draw_networkx_edges(
                grph, pos, width=weights, alpha=0.5, arrows=True
                , connectionstyle='arc3,rad=0.1'
            )
        labs = {x["id"]:x["label"] for x in nodes}
        nx.draw_networkx_labels(
            grph, pos, labels=labs,          # {n: n for n in pos},
            font_size=10, font_family='arial'
        )

    def normalize_weights(self, weights):
        norm_w = [0 if x==0 else 0.1 if x == 1 else round(math.log(x), 3) for x in weights]
        # norm_w = [0.1]*len(weights)
        return norm_w

    def shorten_name(self, url: str):
        parts = url.split("/")
        s = parts[-1]
        return s

def tester():
    from plans.recapi import run_testcase
    testcase = {
        "client_industry_filter": ["Financial Services - WM148049"],
        # "client_region": ["EMEA - WM153003"],
        "recommendation_count": 4,
        "debug": True,
        "algorithm": "algo2"
    }
    node_keys = {
        "url": "document",
        "client_product_filter": "product",
        "client_industry_filter": "industry",
        "client_region_filter": "region",
        "client_country_filter": "country",
        "client_product": "product",
        "client_industry": "industry",
        "client_region": "region",
        "client_country": "country"
    }
    viz = Explainability(0, "name", 0)
    # Data is the docs part of the explainability - list for all recs
    data, start_points = run_testcase(testcase, Explainability.attrib_sep, node_keys)
    # select all recs or any 1 index (this should be <= length of recommendations)
    data_sel = "all"
    dat = viz.combine_data(data) if data_sel == "all" else data[int(data_sel) - 1]
    # 3 means labeling
    grph, nodes, links, aggr_info = viz.controller(dat, start_points, 3)
    # No need for visualisation when only the labeler is required
    # viz.viz_graph(grph, nodes, 3, layout_sel)
    results = viz.get_results(grph, links, aggr_info)
    # Markdown formatted text
    lines = viz.apply_markdown(results)
    # Lines are the output of the labeler in markdown
    # are in order of importance - strong ones top, weaker ones last
    for line in lines:
        print(line)
    return lines
# lines = tester()
