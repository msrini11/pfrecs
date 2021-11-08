# Driver stub to run the Explainability
# Streamlit-based front end , working off a few test cases. Depends on Recapi to run the tests using API
# Parameters can be controlled thru streamlit based UI controls
# Useful to test and play around with parameters
# Usage: python -m streamlit run run_rec.py

import streamlit as st
from explainability import Explainability
import matplotlib.pyplot as plt
from utils import accessProperty
from oce_taxonomies import oce_countries, oce_industries, oce_regions, oce_products
from recapi import run_testcase, get_start_points, run_testcase_oce, run_testcase_algo1, oce_es_rec

@st.cache(allow_output_mutation=True)
def get_result(testcase, mode=2):
    # data, start_map = run_testcase(testcase, Explainability.attrib_sep, node_keys)
    all_rec = oce_es_rec(testcase, Explainability.attrib_sep, node_keys, mode)
    return all_rec

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
class RunRecs:
    def __init__(self, mode=30):
        self.mode = mode

    def controls(self):
        recm = self.init_form()
        data_algo2 = recm.get("pf_algo2")
        data_algo1 = recm.get("pf_algo1")
        oce_recs = recm.get("oce_recs")
        start_points = recm.get("start_points")
        #
        stidx = -1
        st_cols = st.columns(3) if self.mode == 30 \
            else st.columns(2) if self.mode in [6, 15, 10] \
            else st.columns(1)
        if self.mode % 2 == 0: #and data_algo2:
            stidx += 1
            dat, thresh, node_option, nodes_n, cons_sel, l_sel = self.algo2_form(data_algo2)
            with st_cols[stidx]:
                st.markdown('ALGO2')
                visualize_algo2(dat, thresh, start_points, node_option, nodes_n, cons_sel, l_sel)

        if self.mode % 3 == 0:  # and oce_recs:  # OCE
            stidx += 1
            with st_cols[stidx]:
                st.markdown('OCE Rec')
                visualize_oce(oce_recs)

        if self.mode % 5 == 0:  # and data_algo1:
            stidx += 1
            with st_cols[stidx]:
                st.markdown('ALGO1')
                visualize_algo1(data_algo1)

    def algo2_form(self, data_algo2):
        with st.sidebar:
            #with st.form(key='form2'):
            node_type_n = Explainability.node_type_n
            all_node_types = list(Explainability.node_type_c.keys())
            # node_type_sel = st.sidebar.multiselect(
            #     'Entities',
            #     all_node_types,
            #     default=all_node_types,
            #     # on_change=self.node_type_selection,
            #     key="node_type_sel_"
            # )
            node_type_sel = all_node_types
            nodes_n = 1
            for x in node_type_sel:
                nodes_n *= node_type_n.get(x, 1)
            if len(all_node_types) == len(node_type_sel):
                nodes_n = 0

            # label_sel = st.sidebar.selectbox(
            #     'Label Option', ('name', 'id')
            #     # ,key="label_sel_", on_change=self.label_sel
            # )
            label_sel = 'name'
            node_option = label_sel
            opts = ['all']
            if data_algo2:
               opts.extend([str(x + 1) for x in range(len(data_algo2))])
            print(opts)
            data_sel = st.sidebar.radio(
               "Recommended Data algo2", options=opts
            )
            # data_sel = "all"
            # -----------dat, node_option, nodes_n, cons_sel, l_sel------
            viz = Explainability(0, node_option, nodes_n)
            if data_algo2:
                dat = viz.combine_data(data_algo2) if data_sel == "all" else data_algo2[int(data_sel) - 1]
            else:
                dat = {}
            nodes, links = viz.prep(dat)
            if len(nodes) == 0:
                return dat, 0, node_option, nodes_n, None, None
            #
            minw, avgw, maxw, sd = viz.get_avg_weight(links)
            thresh_opt = st.sidebar.slider(
                "Sensitivity Threshold",
                min_value=0.0,
                max_value=round(maxw, 1),
                key="thresh_opt_",
                value=round(minw, 1),
                step=0.1
                # on_change=self.thresh_opt,
            )
            thresh = thresh_opt
            # thresh = 0.0
            print("Self.thresh", thresh)
            #
            layouts = {
                "Circular": "c",
                "Fruchterman Reingold": "f",
                "Spring": "s",
                "Planar": "p"
            }
            # l_sel = 'c'
            layout_sel = st.sidebar.radio(
                "Layout Type",
                options=list(layouts.keys()),
                index=0
            )
            l_sel = layouts.get(layout_sel)

            #
            node_type_cons_options = {
                "Labeler": 3,
                "None": 0,
                "By Node Type": 1,
                "From Start Nodes": 2
            }
            # cons_sel = 3
            cons_sel_ = st.sidebar.radio(
                "Consolidation Type",
                options=list(node_type_cons_options.keys()),
                index=0
            )
            cons_sel = node_type_cons_options.get(cons_sel_)

            # algo2_cfg = st.form_submit_button("Go!")
        return dat, thresh, node_option, nodes_n, cons_sel, l_sel

    def init_form(self):
        with st.sidebar:
            st.markdown('**PathFactory**')
            with st.form(key='rec-params'):
                algo_ids = {'ALGO2': 2, 'OCE Rec': 3, 'ALGO1': 5}
                algos = ['ALGO2', 'OCE Rec', 'ALGO1']
                algos_sel = st.multiselect(
                    "Outputs",
                    algos,
                    default=algos
                )
                algosid = 1
                for sel in algos_sel:
                    algosid *= algo_ids[sel]
                self.mode = algosid
                #
                testcase = {"recommendation_count": 10, "debug": True, "params_must_flag": False}
                inds = st.multiselect(
                    "OCE Industry",
                    oce_industries,
                    default=[]
                )
                testcase["client_industry"] = inds

                prods = st.multiselect(
                    "OCE Products",
                    oce_products,
                    default=[]
                )
                testcase["client_product"] = prods

                regs = st.multiselect(
                    "OCE Regions",
                    oce_regions,
                    default=[]
                )
                testcase["client_region"] = regs

                crys = st.multiselect(
                    "OCE Countries",
                    oce_countries,
                    default=[]
                )
                testcase["client_country"] = crys
                #
                aor_options = ['OR', 'AND']
                aor_sel = st.radio("And or OR", aor_options, index=0)
                if aor_sel == 'OR':
                    params_must_flag = False
                else:
                    params_must_flag = True
                testcase['params_must_flag'] = params_must_flag
                #
                submit = st.form_submit_button("Run Recommendations")
                if len(prods) + len(inds) + len(crys) + len(regs) > 0:
                    recm = get_result(testcase, self.mode)
                else:
                    recm = {}
        return recm

def starter():
    # tcs = [str(idx)+"--"+ ",".join(get_start_node_names(testcases[idx])) for idx in list(range(len(testcases)))]
    mode = 30
    # recm = {}
    # -------



    nodes_n = 0

    # ---------


def visualize_algo1(data):
    if not data:
        return
    for dat in data:
        doc = dat.get('recommended_url')
        titl = dat.get('title')
        contentid = dat.get('content_id')
        bexp = st.expander(doc)
        with bexp:
            st.markdown(f"*{titl}*")
            st.markdown(f"{contentid}")

def visualize_oce(oce_recs):
    if not oce_recs:
        return
    docs = [x.get('slug') for x in oce_recs]
    cnt = 0
    for idx in range(len(docs[:10])):
        cnt += 1
        doc = docs[idx]
        rec = oce_recs[idx]
        bexp = st.expander(f"{cnt}. {doc}") if accessProperty("match", rec, 0) == 0 else st.beta_expander(f"{cnt}*. {doc}")
        kws = accessProperty("fields.keywords", rec, [])
        headline = accessProperty("fields.story_headline", rec, '')
        with bexp:
            st.markdown(f"*{rec.get('name')}*")
            st.write(f"**Keywords:** {kws}")
            st.markdown(f"**Headline:** {headline}")

def visualize_algo2(dat, thresh, start_points, node_option="name", nodes_n=0, cons_sel=0, layout_sel="c"):
    viz = Explainability(thresh, node_option, nodes_n)
    fig = plt.figure()
    fig.suptitle("Algo2 Graphical Model of RandomWalk approach")
    rep_param = {
        'figure.figsize': (12, 9)
    }
    plt.rcParams.update(rep_param)
    # rcParams['figure.figsize'] = 12, 7
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    info = {}
    if dat:
        grph, nodes, links, aggr_info = viz.controller(dat, start_points, cons_sel, layout_sel)
        if cons_sel == 3:
            viz.viz_graph(grph, nodes, cons_sel, layout_sel)
            results = viz.get_results(grph, links, aggr_info)
            doclines_obj = viz.apply_markdown2(results)
            # Lines are the output of the labeler in markdown
            # are in order of importance - strong ones top, weaker ones last
            for doc, doclines in doclines_obj.items():
                bexp = st.expander(doc)
                with bexp:
                    for line in doclines:
                        st.markdown(line)
            # st.pyplot(fig)
        else:
            viz.viz_graph(grph, nodes, cons_sel, layout_sel)
            st.pyplot(fig)
            info = viz.nodes_info(nodes)
            if "topic" in info:
                topics = info["topic"]
                topic_sel = st.sidebar.radio("Topics", options=topics)
                words = info.get("topic-word")

x = RunRecs(30)
x.controls()
