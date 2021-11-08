import requests
from lib.utils import accessProperty

def oce_es_rec(testcase, sep, node_keys, mode=2):
    start_points = get_start_points(testcase, sep, node_keys)
    cnt = 0
    cnt += 1
    # ----
    oce_results = []
    pf_algo2 = []
    pf_algo1 = []
    rcnt = accessProperty('recommendation_count', testcase, 10)
    if mode % 3 == 0:
        # from test case find params
        # 1=>products , 2=>industries, 3=>regions, 4=>countries
        # prod-1, prod-2 * ind-1, ind2 * region-1 * country-1
        # *ind-1
        # **region-1
        # prod-1**region1
        print("-----Running OCE-----")
        prod_s = ",".join(testcase.get("client_product_filter", []) + testcase.get("client_product", []))
        ind_s = ",".join(testcase.get("client_industry_filter", []) + testcase.get("client_industry", []))
        reg_s = ",".join(testcase.get("client_region_filter", []) + testcase.get("client_region", []))
        cty_s = ",".join(testcase.get("client_country_filter", []) + testcase.get("client_country", []))
        params = "*".join([prod_s, ind_s, reg_s, cty_s])
        oce_results, params = oce_read(params)
    if mode % 2 == 0:
        pf_result = run_recapi(testcase, node_keys, "algo2", rcnt)
        pf_algo2 = accessProperty("data", pf_result, [])
    if mode % 5 == 0:
        pf_result = run_recapi(testcase, node_keys, "algo1", rcnt)
        pf_algo1 = accessProperty("data", pf_result, [])
    res = {
        "pf_algo2": pf_algo2,
        "oce_recs": oce_results,
        "entry": testcase,
        "pf_algo1": pf_algo1,
        "start_points": start_points
    }
    return res


def run_recapi(tst, node_keys, default_algo, k=10):
    # url = "http://qa-recommender.pathfactory-development.com:7700/api/v3/relatedcontent"
    url = "http://qa-ci-recommend.pathfactory-development.com:7500/api/v3/ci-recommend/related-content"
    try:
        req = {}
        req.update({
            # "organization_id": "84722990-281f-4235-9553-2bbe88e52608",
            "organization_id": "328aa9e1-0fe0-48ac-8441-5ae8b6c3b336",
            "content_pool_id": "eed1ccb7-0857-4736-869c-253aea4d0e01",
            # "content_pool_id": "00f66f7b-7f42-44e1-b62a-4b7c25b9e514",
            # "recommendation_count": k,
            "k": k,
            "algorithm": default_algo,
            "debug": True,
            "params_must_flag": accessProperty("params_must_flag", tst, False)
        })
        req_filter_keys = [x for x in node_keys if x.endswith("_filter")]
        soft_keys = ["client_product", "client_industry", "client_region", "client_country"]
        for p in tst:
            if p in req_filter_keys:
                req.update({p: tst[p]})
            elif p in soft_keys:
                if "params" not in req:
                    req.update({"params": []})
                req["params"].append({
                    "attribute": p,
                    "values": tst[p] if type(tst[p]) is list else [tst[p]]
                })
        print (req)
        resp = requests.post(url, json=req)
    except Exception as e:
        print(e)
        return {"error": str(e)}
    if not resp.ok:
        return None
    result_obj = resp.json()
    return result_obj


def get_start_points(testcase, sep, node_keys):
    # start_map = {node_keys[k]: [] for k in node_keys.values()}
    start_node_info = []
    for k in node_keys:
        v = testcase.get(k, [])
        vals = v if type(v) is list else [v]
        vals = [sep.join([node_keys[k],str(v)]) for v in vals]
        start_node_info.extend(vals)
        # start_node_names.extend(v if type(v) is list else [v])
    return start_node_info


def run_testcase(testcase, sep, node_keys):
    start_points = get_start_points(testcase, sep, node_keys)
    print ("-----Running Algo2-----")
    rec_obj = oce_es_rec(testcase, node_keys, "algo2", 30)
    pf_result = rec_obj.get("pf_algo2", {})
    data = pf_result.get("docs", [])
    return data, start_points


def run_testcase_algo1(testcase, sep, node_keys):
    # start_points = get_start_points(testcase, sep, node_keys)
    print ("-----Running Algo1-----")
    rec_obj = oce_es_rec(testcase, node_keys, "algo1", 2)
    pf_result = rec_obj.get("pf_result", {})
    return pf_result


def run_testcase_oce(testcase):
    # from test case find params
    # 1=>products , 2=>industries, 3=>regions, 4=>countries
    # prod-1, prod-2 * ind-1, ind2 * region-1 * country-1
    # *ind-1
    # **region-1
    # prod-1**region1
    print("-----Running OCE-----")
    prod_s = ",".join(testcase.get("client_product_filter", []) + testcase.get("client_product", []))
    ind_s = ",".join(testcase.get("client_industry_filter", []) + testcase.get("client_industry", []))
    reg_s = ",".join(testcase.get("client_region_filter", []) + testcase.get("client_region", []))
    cty_s = ",".join(testcase.get("client_country_filter", []) + testcase.get("client_country", []))
    params = "*".join([prod_s, ind_s, reg_s, cty_s])
    items, params = oce_read(params)
    return items
    #{"matches": items, "params": params, "num": len(items)}

def oce_read(params):
    params = extract_params(params)
    base_url = "https://www.oracle.com/node/oce/storyhub/prod/api/v1.1/items"
    token = "562d783bf995ef257c6db4441297cda7"
    ps = 100
    offset = 0
    filters = oce_filter(params)
    q = " AND ".join(filters)
    q = f"({q})"
    items = []
    while (True):
        oce_params = {
            "limit": ps,
            "links": "self",
            "fields": "all",
            # if order by using fields attributes it depends on content type
            "orderBy": "fields.publish_date:desc;name",
            "q": q,
            "offset": offset,
            "channelToken": token,
            "totalResults": True
        }
        resp = requests.get(
            base_url,
            params=oce_params
            # headers={"channelToken": token},
        )
        if resp.status_code != 200:
            break
        resp_obj = resp.json()
        items.extend(resp_obj["items"])
        offset += resp_obj["count"]
        if not resp_obj["hasMore"]:
            break
    # items = self.items_oce_extract(items)
    return items, params

def oce_filter(params):
    content_type = "SH-PublicStory"
    tax_cols = ["countries", "products", "industries", "regions"]
    oce_filters = [
        f'type eq "{content_type}"'
    ]
    for col in tax_cols:
        if col in params and params[col]:
            a = " OR ".join([f'(taxonomies.categories.name eq "{x}")' for x in params[col]])
            oce_filters.append(a)
    return oce_filters

def extract_params(params):
    tax_cols = ["countries", "products", "industries", "regions"]
    # tax_attr_cols = [x for x in list(self.attr_weights.keys()) if x in self.tax_cols]
    if params is None:
        params = {x:[] for x in tax_cols}
    else:
        parts = params.split("*")
        parts = [p.strip() for p in parts]
        params = {}
        # 1=>products , 2=>industries, 3=>regions, 4=>countries
        # prod-1, prod-2 * ind-1, ind2 * region-1 * country-1
        # *ind-1
        # **region-1
        # prod-1**region1
        params["products"] = [] if len(parts) < 1 or parts[0] == "" else parts[0].split(",")
        params["industries"] = [] if len(parts) < 2 or parts[1] == "" else parts[1].split(",")
        params["regions"] = [] if len(parts) < 3 or parts[2] == "" else parts[2].split(",")
        params["countries"] = [] if len(parts) < 4 or parts[3] == "" else parts[3].split(",")
        #
        params["countries"] = [x.strip() for x in params["countries"]]
        params["industries"] = [x.strip() for x in params["industries"]]
        params["products"] = [x.strip() for x in params["products"]]
        params["regions"] = [x.strip() for x in params["regions"]]

        params["countries"].sort()
        params["industries"].sort()
        params["products"].sort()
        params["regions"].sort()
    # ws = [len(params.get(w,""))*self.attr_weights[w] for w in self.attr_weights]
    # self.total_weight = sum(ws)*1.0
    return params
