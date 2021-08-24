from opcua import Server


def setupOPCServer(endpoint="opc.tcp://127.0.0.1:5000"):
    server = Server()
    server.set_endpoint(endpoint)

    ### Register NameSpace
    namespace = server.register_namespace("Ladles")
    node = server.get_objects_node()
    # print(objects)
    # ladlesOPCObj = node.add_object('ns=2; s="Ladle Number"','Ladle Numbers')
    ladlesOPCObj = node.add_object(namespace, 'Ladle Numbers')
    ladleLeftOPCVar = ladlesOPCObj.add_variable(namespace, "Left Ladle No", 0)
    ladleRightOPCVar = ladlesOPCObj.add_variable(namespace, "Right Ladle No", 0)

    print("Starting OPC Server...")
    server.start()
    print("OPC-UA Server Online")
    return server

def stopOPCServer():
    server = Server()
    print(server.get_endpoints())
