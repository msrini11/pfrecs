import os
def accessProperty(prop, mdata, default_value=None):
    '''
    Return the property present in mdata which is the object that potentially contains prop
    prop can be a compound element for accessing deeper elements, which are connected
    thru hash structures
    Array structures cannot be resolved so they are not permitted. Returns None or default_value
    if None or cannot find
    :param prop:
    :param mdata:
    :return: the leaf element, or compound object or None
    '''
    if mdata is None:
        return default_value
    if prop in mdata:
        val = default_value if mdata[prop] is None else mdata[prop]
        return val  #mdata[prop]      #could be a leaf element or compound object
    if isinstance(prop, str) and "." in prop:  # compound element
        props = prop.split('.')
        elem_value = mdata
        for prs in props:
            if elem_value is None:
                return default_value
            if prs in elem_value:
                elem_value = elem_value[prs]
            else:
                return default_value
        #fld = props[-1]  # retain the last one as the field name
        val = default_value if elem_value is None else elem_value
        return val      #elemVal
    return default_value


def get_env_variable(var_name, default=None):
    # Get Environment variable, if not set, return the default
    try:
        return os.environ[var_name]
    except KeyError:
        return default


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def mongo2snowflake_class_def(mongo_def):
    fld_type_map = mongo_def._fields
    lines = []
    type_map = {
        "DateTimeField": "DATETIME",
        "StringField": "STRING",
        "FloatField": "FLOAT",
        "IntField": "INT",
        "DateField": "DATE",
        "ObjectIdField": "STRING",
        "EmbeddedDocumentField": "OBJECT",
        "ListField": "ARRAY",
        "ImageField": "BINARY"
    }
    line1 = f"self.{mongo_def.__name__} = Table("
    lines.append(line1)
    line2 = f"'{mongo_def.__name__}', self.alm,"
    lines.append(line2)
    for fld in fld_type_map:
        for typ in type_map:
            if typ in str(fld_type_map[fld]):
                line = f"Column('{fld}', {type_map[typ]}),"
                lines.append(line)
                break
    lines.append(")")
    return lines
