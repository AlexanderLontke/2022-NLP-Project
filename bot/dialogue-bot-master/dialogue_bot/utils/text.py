import itertools
import random

from typing import List, Dict, Union


def replace_spans(
    inp: Union[str, list],
    replacements: List[dict],
    value_func=lambda x: x,
    else_func=lambda x: x,
    random_mode: bool = False,
    shuffle: bool = True,
    max_realizations: int = None,
    output_replacements: bool = False,
):
    """
    Replaces specified spans of a text or a list with the given replacements.
    :param inp: Text or list
    :param replacements: A list of dictionaries which need to have the following keys:
        - "start": The start position of the span that should be replaced
        - "end": The end position of the span that should be replaced (== start+length)
        - "values": A value that will be used to replace the span /
                    or a list of value-options where one of them is selected to replace the span
    :param value_func: A function that will be applied to convert the chosen value into the new span
    :param else_func: A function that will be applied to convert the non-replacing parts into new spans
        (can be used to regex-escape). This will also be applied if len(replacements) == 0.
    :param random_mode: Randomly chooses from value-options to replace the span.
        Benefit of this is runtime if there are a lot of value-permutations to choose from
        ATTENTION: this may cause duplicate outputs
    :param shuffle: Determine whether the output-realizations are shuffled
    :param max_realizations: Limit the maximum amount of generated outputs
    :param output_replacements: Set to True if you want to additionally output the replacement-dictionaries that are
        enriched with the following information:
        -"res_start": The resulting start position of the replaced span
        -"res_end": The resulting end position of the replaced span
        -"res_value": The chosen value which was used to replace the span
        -"res_span": The new span that was generated by applying **value_func** to **res_value*
    :return: The replaced text or list (with replacement-dictionaries if **output_replacements** is True)
    """

    def mult(numbers):
        res = 1
        for n in numbers:
            res *= n
        return res

    def replacement_values(replacement: dict):
        v = replacement["values"]
        return [v] if not isinstance(v, list) else v

    def split_to_parts(replacements: List[dict]):
        """
        Convert inp into parts.
        Example: "hello XXX you" => [(False, "hello "), (True, 0), (False, " you")]
        """

        id_replacements = enumerate(replacements)  # give replacements an id
        id_replacements = sorted(
            id_replacements, key=lambda t: (t[1]["start"], -t[1]["end"])
        )  # sort by 'start'

        parts = []
        remaining_inp = inp
        offset = 0
        for i, dct in id_replacements:
            start, end = dct["start"] + offset, dct["end"] + offset
            if start < 0:
                continue  # overlap

            before, extr, after = (
                remaining_inp[:start],
                remaining_inp[start:end],
                remaining_inp[end:],
            )

            # convert before into part
            parts.append((False, before))

            # convert extraction into part
            parts.append((True, i))

            remaining_inp = after
            offset -= end

        # convert after into part
        parts.append((False, remaining_inp))
        return parts

    def realize(
        replacements: List[dict],
        parts: List[tuple],
        repl_values: Dict[str, int],
        value_func,
        else_func,
        output_replacements: bool,
    ):
        res = None
        res_replacements = []
        start = 0
        for (is_repl, x) in parts:
            if is_repl:
                replacement = replacements[x]
                value = replacement_values(replacement)[repl_values[x]]
                span = value_func(value)

                res_replacement = {k: v for k, v in replacement.items()}
                res_replacement["res_start"] = start  # add additional attribute
                res_replacement["res_end"] = start + len(
                    span
                )  # add additional attribute
                res_replacement["res_value"] = value  # add additional attribute
                res_replacement["res_span"] = span  # add additional attribute
                res_replacements.append(res_replacement)
            else:
                span = else_func(x)

            if res is None:
                res = span
            else:
                res += span
            start += len(span)

        if output_replacements:
            return res, res_replacements
        else:
            return res

    parts = split_to_parts(replacements)
    used_replacement_idxs = [i for (b, i) in parts if b]
    nr_realizations = mult(
        [len(replacement_values(replacements[i])) for i in used_replacement_idxs]
    )

    res = []
    if random_mode:
        # random realizations (this can cause duplications!)
        for e in range(nr_realizations):
            if (max_realizations is not None) and (max_realizations == e):
                break
            repl_values = {
                i: random.choice(range(len(replacement_values(replacements[i]))))
                for i in used_replacement_idxs
            }
            res.append(
                realize(
                    replacements,
                    parts,
                    repl_values,
                    value_func,
                    else_func,
                    output_replacements,
                )
            )
    else:
        # deterministic realizations
        for e, value_idxs in enumerate(
            itertools.product(
                *[
                    range(len(replacement_values(replacements[i])))
                    for i in used_replacement_idxs
                ]
            )
        ):
            if (
                (max_realizations is not None)
                and (max_realizations == e)
                and (not shuffle)
            ):
                break
            repl_values = {i: j for i, j in zip(used_replacement_idxs, value_idxs)}
            res.append(
                realize(
                    replacements,
                    parts,
                    repl_values,
                    value_func,
                    else_func,
                    output_replacements,
                )
            )

        if shuffle:
            random.shuffle(res)

        if max_realizations is not None:
            res = res[:max_realizations]

    return res


if __name__ == "__main__":
    for x in replace_spans(
        "01234567890",
        [
            {"start": 0, "end": 0 + 2, "values": "XXXX"},
            {"start": 5, "end": 5 + 2, "values": ["WWW", "ZZZ"]},
        ],
        output_replacements=True,
    ):
        print(x)
