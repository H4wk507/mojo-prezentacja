from max.graph import Graph, TensorType, Type
from max import engine
from tensor import Tensor, TensorShape
from random import rand


def main():
    graph = Graph(
        in_types=List[Type](
            TensorType(DType.float32, 512, 512),
            TensorType(DType.float32, 512, 512),
        )
    )
    out = graph[0] @ graph[1]
    graph.output(out)
    graph.verify()

    session = engine.InferenceSession()
    model = session.load(graph)

    var input0 = Tensor[DType.float32](TensorShape(512, 512))
    rand(input0.unsafe_ptr(), input0.num_elements())
    var input1 = Tensor[DType.float32](TensorShape(512, 512))
    rand(input1.unsafe_ptr(), input1.num_elements())

    for i in range(1):
        var ret = model.execute("input0", input0, "input1", input1)
