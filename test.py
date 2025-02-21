import json
import numpy as np
import array
import struct
import argparse
import sys


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size
        )
        self.weights_hidden_hidden = np.random.randn(
            self.hidden_size, self.hidden_size
        )
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size
        )
        self.bias_hidden_ih = np.zeros((1, self.hidden_size))

        self.bias_hidden_hh = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return x * (x > 0)

    def relu_derivative(self, x):
        return 1.0 * (x > 0)

    def feedforward(self, X):
        self.hidden_activation_ih = (
            np.dot(X, self.weights_input_hidden) + self.bias_hidden_ih
        )
        self.hidden_output_ih = self.sigmoid(self.hidden_activation_ih)

        self.hidden_activation_hh = (
            np.dot(self.hidden_output_ih, self.weights_hidden_hidden)
            + self.bias_hidden_hh
        )
        self.hidden_output_hh = self.sigmoid(self.hidden_activation_hh)

        self.output_activation = (
            np.dot(self.hidden_output_hh, self.weights_hidden_output)
            + self.bias_output
        )
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(
            self.predicted_output
        )

        hidden_error_hh = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta_hh = hidden_error_hh * self.sigmoid_derivative(
            self.hidden_output_hh
        )

        hidden_error_ih = np.dot(hidden_delta_hh, self.weights_hidden_hidden.T)
        hidden_delta_ih = hidden_error_ih * self.sigmoid_derivative(
            self.hidden_output_ih
        )

        # Update weights and biases
        self.weights_hidden_output += (
            np.dot(self.hidden_output_hh.T, output_delta) * learning_rate
        )
        self.bias_output += (
            np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        )
        self.weights_hidden_hidden += (
            np.dot(self.hidden_output_ih.T, hidden_delta_hh) * learning_rate
        )
        self.bias_hidden_hh += (
            np.sum(hidden_delta_hh, axis=0, keepdims=True) * learning_rate
        )
        self.weights_input_hidden += (
            np.dot(X.T, hidden_delta_ih) * learning_rate
        )
        self.bias_hidden_ih += (
            np.sum(hidden_delta_ih, axis=0, keepdims=True) * learning_rate
        )

    def set_params(self, params: dict):
        params_sz = 0
        self.weights_input_hidden = np.asarray(params.get("w1"))
        params_sz += self.weights_input_hidden.size
        self.bias_hidden_ih = np.asarray(params.get("b1"))
        params_sz += self.bias_hidden_ih.size
        self.weights_hidden_hidden = np.asarray(params.get("w2"))
        params_sz += self.weights_hidden_hidden.size
        self.bias_hidden_hh = np.asarray(params.get("b2"))
        params_sz += self.bias_hidden_hh.size
        self.weights_hidden_output = np.asarray(params.get("w3"))
        params_sz += self.weights_hidden_output.size
        self.bias_output = np.asarray(params.get("b3"))
        params_sz += self.bias_output.size
        print(f"[*] Loaded {params_sz} parameters")

    def train(self, X, y, epochs, learning_rate):
        last_loss = 0
        loss = 1
        try:
            for epoch in range(epochs):
                output = self.feedforward(X)
                self.backward(X, y, learning_rate)
                if epoch % 1000 == 0:
                    loss = np.mean(np.square(y - output))
                    print(
                        f"Epoch {epoch:>20}, Loss:{np.round(loss,decimals=10):>13} [D={np.round(loss - last_loss, decimals=7):+8}]"
                    )
                    last_loss = loss
                if loss < 0.008:
                    break
        except KeyboardInterrupt:
            print("Breaking training")
        return self.todict()

    def todict(self):
        return {
            "w1": self.weights_input_hidden.tolist(),
            "b1": self.bias_hidden_ih.tolist(),
            "w2": self.weights_hidden_hidden.tolist(),
            "b2": self.bias_hidden_hh.tolist(),
            "w3": self.weights_hidden_output.tolist(),
            "b3": self.bias_output.tolist(),
        }

    def set_frombin(self, path: str):
        def decode_header(data: bytes):
            layers_count = struct.unpack("<h", data[:2])
            weights_fmt = "hh" * layers_count[0]
            biases_fmt = "hh" * layers_count[0]
            dcdd = struct.unpack("<h" + weights_fmt + biases_fmt, data)
            return dcdd

        def read_message(data: bytes):
            header_sz = struct.unpack("<h", data[:2])[0]
            header_data = decode_header(data[2 : header_sz + 2])
            wts = [0] * header_data[0]
            bss = [0] * header_data[0]
            blocksz_i = 1
            start = header_sz + 2
            obj = {}
            for i in range(header_data[0]):
                wts[i] = {}
                block_sz = header_data[blocksz_i] * header_data[blocksz_i + 1]
                wts[i]["info"] = [
                    header_data[blocksz_i],
                    header_data[blocksz_i + 1],
                ]
                wts[i]["data"] = array.array("d")
                wts[i]["data"].frombytes(data[start : block_sz * 8 + start])
                blocksz_i += 2
                start += block_sz * 8
            for i in range(header_data[0]):
                bss[i] = {}
                block_sz = header_data[blocksz_i] * header_data[blocksz_i + 1]
                bss[i]["info"] = [
                    header_data[blocksz_i],
                    header_data[blocksz_i + 1],
                ]
                bss[i]["data"] = array.array("d")
                bss[i]["data"].frombytes(data[start : block_sz * 8 + start])
                blocksz_i += 2
                start += block_sz * 8
            for i in range(header_data[0]):
                obj["w" + str(i + 1)] = []
                k = 0
                j = 0
                while j < len(wts[i]["data"]):
                    j += wts[i]["info"][1]
                    obj["w" + str(i + 1)].append(wts[i]["data"][k:j].tolist())
                    k = j
                obj["b" + str(i + 1)] = []
                k = 0
                j = 0
                while j < len(bss[i]["data"]):
                    j += bss[i]["info"][1]
                    obj["b" + str(i + 1)].append(bss[i]["data"][k:j].tolist())
                    k = j
            return obj

        data = None
        with open(path, "rb") as f:
            data = f.read()
        self.set_params(read_message(data))

    def binarize(self):
        data = self.todict()
        weights = array.array("d")
        biases = array.array("d")
        head_w = b""
        head_b = b""
        n_layers = 0
        for key in data:
            val = data[key]
            if key.startswith("w"):
                weight = array.array("d")
                head_w += struct.pack("<hh", len(val), len(val[0]))
                for nweight in val:
                    weight.extend(array.array("d", nweight))
                weights.extend(weight)
            if key.startswith("b"):
                bias = array.array("d")
                head_b += struct.pack("<hh", len(val), len(val[0]))
                for nbias in val:
                    bias.extend(array.array("d", nbias))
                biases.extend(bias)
                n_layers += 1

        header_data = (
            struct.pack("<h", n_layers) + head_w + struct.pack("<") + head_b
        )
        _bin = (
            struct.pack("<h", len(header_data))
            + header_data
            + weights.tobytes()
            + biases.tobytes()
        )
        return _bin

    def tobin(self, path: str):
        with open(path, "wb") as f:
            f.write(self.binarize())


def format_data(data):
    inputs = []
    expected = []
    for d in data:
        lvl = d.get("level")
        t = d.get("total_seconds") / (60 * 60 * 24 * 28)
        skills = np.array(d.get("skills")) / 20
        exams = np.array(d.get("exams")) / 100
        rushes = np.array(d.get("rushes")) / 125
        projects = np.array(d.get("projects")) / 100
        _in = [lvl / 21, t, np.sum(projects * 100) / (len(projects) * 100)]
        _in = np.concatenate((_in, skills, rushes, projects, exams))

        inputs.append(_in)
        expected.append(
            [int(not d.get("passed", False)), int(d.get("passed", False))]
        )

    return np.array(inputs), np.array(expected)


def test_obj_from_input():
    print("[+] Enter the required values from your C Piscine Cursus:")
    lvl = float(input("Level> "))
    print("[?] Enter your logtime in seconds (seconds = hours * 3600)")
    t = float(input("Logtime> "))
    print("[?] Enter your skills in this specific order:")
    print("\tAlgo/AI|Rigor|Unix|Adaptation")
    print("\tInput it as in your intra")
    skills = [float(s) for s in input("Skills> ").split("|")]
    print(
        "[!] IMPORTANT: For the following data, "
        "input '-1' if you did not participate to it."
    )
    print("[?] Enter your rushes marks in this specific order:")
    print("\tRush00|Rush01|Rush02")
    rushes = [int(r) for r in input("Rushes> ").split("|")]
    print("[?] Enter your exams results in this specific order:")
    print("\tExam00|Exam01|Exam02|FinalExam")
    exams = [int(e) for e in input("Exams> ").split("|")]
    print("[?] Enter your days results in this specific order:")
    print(
        "\tShell00|Shell01|C00|C01|C02|C03|C04|C05|C06"
        "|C07|C08|C09|C10|C11|C12|C13|BSQ"
    )
    projects = [int(p) for p in input("Projects> ").split("|")]
    return [
        {
            "level": lvl,
            "total_seconds": t,
            "skills": skills,
            "rushes": rushes,
            "exams": exams,
            "projects": projects,
        }
    ]


def test(test_data):
    nn = NeuralNetwork(input_size=31, hidden_size=64, output_size=2)
    nn.set_frombin("params.bin")

    def predict(output: list[float]):
        if output[0] > output[1]:
            return 0
        return 1

    output = nn.feedforward(test_data[0])
    for i in range(len(output)):
        print(
            f"Prediction: {'Accepted' if predict(output[i]) else 'Refused'} "
            f"({max(output[i] * 100):.5}%)"
        )


if __name__ == "__main__":
    desc_txt = """Test your result at the piscine!

Every parameter must be set in the precise specified order, as it is the same as the model was trained on.
If you are not using the interactive mode, each missing value will be set to -1.
You do not need to input anything for the projects you did not subscribe to.
If you desire, you may put '-1' as a place holder value in such cases."""
    parser = argparse.ArgumentParser(
        prog="pisciNNe",
        description=desc_txt,
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=True,
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Use the interactive mode",
    )
    parser.add_argument(
        "--level", type=float, help="Your level, as a float", default=0
    )
    parser.add_argument(
        "--logtime",
        type=float,
        help="Your logtime in seconds (seconds = hours * 3600)",
        default=0,
    )
    parser.add_argument(
        "--skill",
        nargs="*",
        type=float,
        help="Your skill levels, as floats, "
        "in the order: ALGO RIGOR UNIX ADAPTATION",
        default=[0, 0, 0, 0],
        metavar="level",
    )
    parser.add_argument(
        "--rushes",
        nargs="*",
        type=int,
        default=[-1, -1, -1],
        metavar="score",
        help="Your scores to the rushes, from 0 to 125, "
        "in the order: RUSH00 RUSH01 RUSH02",
    )
    parser.add_argument(
        "--exams",
        nargs="*",
        type=int,
        metavar="score",
        help="Your scores to the exams, from 0 to 100, "
        "in the order: EXAM00 EXAM01 EXAM02 FINALEXAM",
        default=[-1, -1, -1, -1],
    )
    parser.add_argument(
        "--shell",
        nargs="*",
        type=int,
        metavar="score",
        help="Your scores to the shell projects, from 0 to 100, "
        "in the order: SHELL00 SHELL01",
        default=[-1, -1],
    )
    parser.add_argument(
        "--c",
        nargs="*",
        type=int,
        metavar="score",
        help="Your scores to the C projects, from 0 to 100, "
        "in the order: C00 C01 C02 ... C11 C12 C13",
        default=[-1] * 14,
    )
    parser.add_argument(
        "--bsq",
        type=int,
        default=-1,
        metavar="score",
        help="Your score to the BSQ, from 0 to 100",
    )

    args = parser.parse_args()
    if args.interactive:
        test(format_data(test_obj_from_input()))
        exit(0)
    slvl: list[float] = args.skill
    if len(slvl) < 4:
        diff = 4 - len(slvl)
        print(
            f"[!] Warning, {diff} "
            f"missing skill value{'s' if diff > 1 else ''} "
            "have been defaulted to 0."
        )
        slvl += [0] * diff

    rscores: list[float] = args.rushes
    if len(rscores) < 3:
        diff = 3 - len(rscores)
        print(
            f"[!] Warning, {diff} "
            f"missing rush score{'s' if diff > 1 else ''} "
            "have been defaulted to -1."
        )
        rscores += [-1] * diff

    escores: list[float] = args.exams
    if len(escores) < 4:
        diff = 4 - len(escores)
        print(
            f"[!] Warning, {diff} "
            f"missing exam score{'s' if diff > 1 else ''} "
            "have been defaulted to -1."
        )
        escores += [-1] * diff

    shscores: list[float] = args.shell
    if len(shscores) < 2:
        diff = 2 - len(shscores)
        print(
            f"[!] Warning, {diff} "
            f"missing shell project score{'s' if diff > 1 else ''} "
            "have been defaulted to -1."
        )
        shscores += [0] * diff

    cscores: list[float] = args.c
    if len(cscores) < 14:
        diff = 14 - len(cscores)
        print(
            f"[!] Warning, {diff} "
            f"missing c project score{'s' if diff > 1 else ''} "
            "have been defaulted to -1."
        )
        cscores += [-1] * diff
    print("[*] Data for prediction:")
    print(f"[+] Level:\t{args.level:< 8.2f}")
    print(f"[+] Logtime:\t{args.logtime:< 10.3f}s")
    print("[+] Skills:\t" + "".join([f"{lvl:< 4.2f}" for lvl in slvl]))
    print("[+] Rushes:\t" + "".join([f"{s:< 5}" for s in rscores]))
    print("[+] Exams:\t" + "".join([f"{s:< 5}" for s in escores]))
    print("[+] Shell:\t" + "".join([f"{s:< 5}" for s in shscores]))
    print("[+] C Days:\t" + "".join([f"{s:< 5}" for s in cscores]))
    print(f"[+] BSQ:\t{args.bsq:< 4}")

    obj = [
        {
            "level": args.level,
            "total_seconds": args.logtime,
            "skills": slvl,
            "rushes": rscores,
            "exams": escores,
            "projects": shscores + cscores + [args.bsq],
        }
    ]
    test(format_data(obj))
