import codecs
from io import StringIO

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import pickle


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


class PieceWiseLinearRegressionTree:
    """Regression tree that uses linear regression equations as response generators.

    Args:
        X (array like [2D]): Matrix of independent variables.
        y (array like): Vector of dependent variables
        min_samples_split ([float, int], optional): Fraction (or number) of samples
            required to be in a each child node for a split to be valid.
            Defaults to 0.05.
        max_depth (int, optional): Maximum allowable tree depth. Defaults to 4.
        tree_vars (array like, optional): Variables to be considered when
            generating splits for the PLRT. If None, all variables from `X`
            are used. Variables must be in `X`.
            Defaults to None.
        reg_vars (array like, optional): Variables to use when fitting regression
            equations within each leaf. If None, all variable form `X` are used.
            Variables must be in `X`.
            Defaults to None.
        n_disc_steps (int, optional): Number of discretization samples used for
            determining the optimal threshold for each split. Higher numbers
            may result in a more optimal split
            but at the cost of performance. Defaults to 1000.
        method (str, optional): Currently not used. In future will determine
            optimzation method for determining thresholds for each split.
            Defaults to "exhaustive".
        feature_names (array like, optional): List of feature names to be used
            when exporting or displaying the tree. Defaults to None.
        response_name (str, optional): Name of response variable to be used when
            exporting or displaying the tree. Defaults to None.
        njobs (int, optional): Number of parallel jobs to use when finding optimal
            splits. Defaults to 1.
        _curr_depth (int, optional): INTERNAL USE ONLY: Current depth of tree.
            Defaults to 0.
        _node_type ([str, None], optional): INTERNAL USE ONLY: Type of current node
            [left, right]. Defaults to None.
        _ID (int, optional): INTERNAL USE ONLY: current node ID. Defaults to 0.
        _parent ([int, None], optional): INTERNAL USE ONLY: node parent ID.
            Defaults to None.
        _vars_indices_already (bool, optional): INTERNAL USE ONLY: indicates if
            reg/tree vars have already been converted to indices of self.feats. 
    """

    node_count = 0
    init_N = 0
    min_mse = float("inf")
    max_mse = -float("inf")

    def __init__(
        self,
        X,
        y,
        min_samples_split=0.05,
        max_depth=4,
        tree_vars=None,
        reg_vars=None,
        n_disc_steps=1000,
        method="exhaustive",
        feature_names=None,
        response_name=None,
        njobs=1,
        # variables used internally
        _curr_depth=0,
        _node_type=None,
        _ID=0,
        _parent=None,
        _vars_indices_already=False
    ):
        self.N = X.shape[0]
        if isinstance(X, pd.DataFrame):
            self.X = X.values
            self.feats = list(X.columns)
        else:
            self.X = X
            self.feats = list(range(X.shape[1]))

        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.y = y.values
            self.response = y.name
        else:
            self.y = y
            self.response = "y"

        self.feats = feature_names if feature_names else self.feats
        self.response = response_name if response_name else self.response

        if tree_vars is not None:
            if _vars_indices_already:
                self.tree_vars = tree_vars
            else:
                self.tree_vars = [self.feats.index(i) for i in tree_vars]
        else:
            self.tree_vars = list(range(X.shape[1]))

        if reg_vars is not None:
            if _vars_indices_already:
                self.reg_vars = reg_vars
            else:
                self.reg_vars = [self.feats.index(i) for i in reg_vars]
        else:
            self.reg_vars = list(range(X.shape[1]))

        if isinstance(min_samples_split, int):
            self.min_samples_split = min_samples_split
        elif isinstance(min_samples_split, float):
            self.min_samples_split = int(self.N * min_samples_split)
        else:
            self.min_samples_split = int(self.N * 0.05)

        self.n_disc_steps = n_disc_steps
        self.max_depth = max_depth if max_depth else 4
        self._curr_depth = _curr_depth
        self.method = method

        self.njobs = njobs

        self.rule = None

        self._node_type = _node_type if _node_type else "root"
        self._ID = _ID
        self._parent = _parent
        if _ID == 0:
            PieceWiseLinearRegressionTree.node_count = 0
            PieceWiseLinearRegressionTree.min_mse = float("inf")
            PieceWiseLinearRegressionTree.max_mse = -float("inf")
            PieceWiseLinearRegressionTree.init_N = self.N

        self.left = None
        self.right = None

        self.best_feat = None
        self.best_val = None

        try:
            self.params = self._solve_regression()
        except np.linalg.LinAlgError as e:
            print(f"Cannot solve initial regression due to {e}")

        yhat = self._predict_regression(self.params)

        self.mse = mean_squared_error(self.y, yhat)
        if self.mse < PieceWiseLinearRegressionTree.min_mse:
            PieceWiseLinearRegressionTree.min_mse = self.mse
        if self.mse > PieceWiseLinearRegressionTree.max_mse:
            PieceWiseLinearRegressionTree.max_mse = self.mse

    def _solve_regression(self, X=None, y=None, reg_vars=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        reg_vars = self.reg_vars if reg_vars is None else reg_vars
        X = X[:, reg_vars]
        return np.linalg.inv(X.T @ X) @ (X.T @ y)

    def _predict_regression(self, p, X=None):
        X = self.X if X is None else X
        X = X[:, self.reg_vars]
        return X @ p

    def _split_node_data(self, thresh, feat_id, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        right = np.argwhere(X[:, feat_id] > thresh)
        left = np.argwhere(X[:, feat_id] <= thresh)

        X_right = X[right[:, 0]]
        X_left = X[left[:, 0]]
        y_right = y[right[:, 0]]
        y_left = y[left[:, 0]]

        return X_left, X_right, y_left, y_right

    def _check_all_entries_same(self, matrix):
        bad_columns = []
        for col in range(matrix.shape[1]):
            result = np.all(matrix[:, col] == matrix[0, col])
            if result:
                bad_columns.append(col)
        return bad_columns

    def _check_all_entries_zero(self, matrix):
        return list(np.where(~matrix.any(axis=0))[0])

    def _get_node_score(self, thresh, feat_id, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        X_left, X_right, y_left, y_right = self._split_node_data(thresh, feat_id)

        N_left = y_left.shape[0]
        N_right = y_right.shape[0]

        if N_left == 0 or N_right == 0:
            return (np.nan, "error")
        if N_left < self.min_samples_split or N_right < self.min_samples_split:
            return (np.nan, "error")
        try:
            p_left = self._solve_regression(X_left, y_left)
        except np.linalg.LinAlgError:
            return (np.nan, "error")

        try:
            p_right = self._solve_regression(X_right, y_right)
        except np.linalg.LinAlgError:
            return (np.nan, "error")

        # calc regression score
        yhat_left = self._predict_regression(p_left, X_left)
        yhat_right = self._predict_regression(p_right, X_right)

        mse_left = mean_squared_error(y_left, yhat_left)
        mse_right = mean_squared_error(y_right, yhat_right)

        left_score = N_left / y.shape[0] * mse_left
        right_score = N_right / y.shape[0] * mse_right

        reg_score = left_score + right_score

        # calc persistance score
        pre_index = self.feats.index("release_pre")

        yhat_left = X_left[:, pre_index]
        yhat_right = X_right[:, pre_index]

        mse_left = mean_squared_error(y_left, yhat_left)
        mse_right = mean_squared_error(y_right, yhat_right)

        left_score = N_left / y.shape[0] * mse_left
        right_score = N_right / y.shape[0] * mse_right

        pers_score = left_score + right_score
        tolerance = 0.10
        diff = (pers_score - reg_score) / pers_score
        if diff < tolerance:
            return (pers_score, "pers")
        else:
            return (reg_score, "reg")

    def _get_best_thresh_var(self, feat_id, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        # np.unique returns the sorted unique entries
        # [1:-1] removes the last two and first two thresholds which cannot be used
        thresh_possib = np.linspace(
            X[:, feat_id].min(), X[:, feat_id].max(), self.n_disc_steps
        )
        thresh_possib = thresh_possib[1:-1]
        scores = Parallel(n_jobs=-1, verbose=0)(
            delayed(self._get_node_score)(t, feat_id, X, y) for t in thresh_possib
        )
        try:
            best = np.nanargmin([i[0] for i in scores])
        except ValueError:
            # all nans in scores
            return np.nan, np.nan, "error"
        return thresh_possib[best], *scores[best]

    def _optimize_node(self, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        best_feat = None
        best_val = None
        mse = self.mse

        if self.njobs > 1:
            opts = [self._get_best_thresh_var(i) for i in self.tree_vars]
            try:
                best_idx = np.nanargmin([i[1] for i in opts])
            except ValueError:
                # all nans in opts
                return best_feat, best_val
            best = self.tree_vars[best_idx]
            opt = opts[best_idx]
            split_type = opt[2]
            # if split_type == "error":
            #     # here, all the mses are infinity
            #     return best_feat, best_val
            print(f"Optimal Split: {self.feats[best]} <= {opt[0]:.3f} - {split_type}")

            if opt[1] < mse:
                X_left, X_right, y_left, y_right = self._split_node_data(opt[0], best)
                if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                    best_feat = best
                    best_val = opt[0]
                    mse = opt[1]
                    if mse < PieceWiseLinearRegressionTree.min_mse:
                        PieceWiseLinearRegressionTree.min_mse = mse
                    if mse > PieceWiseLinearRegressionTree.max_mse:
                        PieceWiseLinearRegressionTree.max_mse = mse
        else:
            for feat_id in self.tree_vars:
                opt = self.get_best_thresh_var(feat_id)

                if opt[1] < mse:
                    X_left, X_right, y_left, y_right = self._split_node_data(
                        opt[0], feat_id
                    )
                    if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                        best_feat = feat_id
                        best_val = opt[0]
                        mse = opt[1]
                        if mse < PieceWiseLinearRegressionTree.min_mse:
                            PieceWiseLinearRegressionTree.min_mse = mse
                        if mse > PieceWiseLinearRegressionTree.max_mse:
                            PieceWiseLinearRegressionTree.max_mse = mse

        return best_feat, best_val

    def fit(self):
        """Fit the model defined by this class"""
        N = self.X.shape[0]

        if (self._curr_depth < self.max_depth) and (N >= self.min_samples_split):
            best_feat, best_val = self._optimize_node()
            if best_feat is not None:
                X_left, X_right, y_left, y_right = self._split_node_data(
                    best_val, best_feat
                )

                self.best_feat = best_feat
                self.best_val = best_val
                self.rule = f"{self.feats[best_feat]} &le; {best_val:.3f}"

                left = PieceWiseLinearRegressionTree(
                    X=X_left,
                    y=y_left,
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    _curr_depth=self._curr_depth + 1,
                    method=self.method,
                    feature_names=self.feats,
                    response_name=self.response,
                    tree_vars=self.tree_vars,
                    reg_vars=self.reg_vars,
                    njobs=self.njobs,
                    n_disc_steps=self.n_disc_steps,
                    _node_type="left_node",
                    _ID=PieceWiseLinearRegressionTree.node_count + 1,
                    _parent=self._ID,
                    _vars_indices_already=True,
                )

                PieceWiseLinearRegressionTree.node_count += 1
                self.left = left
                self.left.fit()

                right = PieceWiseLinearRegressionTree(
                    X=X_right,
                    y=y_right,
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    _curr_depth=self._curr_depth + 1,
                    method=self.method,
                    feature_names=self.feats,
                    response_name=self.response,
                    tree_vars=self.tree_vars,
                    reg_vars=self.reg_vars,
                    njobs=self.njobs,
                    n_disc_steps=self.n_disc_steps,
                    _node_type="right_node",
                    _ID=PieceWiseLinearRegressionTree.node_count + 1,
                    _parent=self._ID,
                    _vars_indices_already=True,
                )

                PieceWiseLinearRegressionTree.node_count += 1
                self.right = right
                self.right.fit()

    def _print_info(self, width=4):
        const = int(self._curr_depth * width**1.5)
        spaces = "-" * const

        if self._node_type == "root":
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {self.mse:.2f}")
        print(f"{' ' * const}   | N obs in the node: {self.X.shape[0]:.0f}")

    def _print_params(self, width=4):
        const = int(self._curr_depth * width**1.5)
        param_format = [
            f"{self.feats[i]}: {j:.3f}" for i, j in zip(self.reg_vars, self.params)
        ]
        print(f"{' ' * const}   | Regression Params:")
        for p in param_format:
            print(f"{' ' * (const + width)}   | {p}")

    def print_tree(self):
        """Print tree from fitted model."""
        self._print_info()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()

        if self.left is None and self.right is None:
            self._print_params()

    def _find_params(self, row, path):
        path.append(self._ID)
        if self.best_feat is not None:
            if row[self.best_feat] <= self.best_val:
                return self.left._find_params(row, path)
            else:
                return self.right._find_params(row, path)
        else:
            return (self.params, self._ID, path)

    def apply(self, X=None):
        """Apply fitted model to fitted data or new data.

        Args:
            X (array like [2D], optional): Independent variables to use
                when applying the tree. If None, the `X` used to fit the
                model is used. Defaults to None.

        Returns:
            tuple: parameters, node id, and paths to get to that node
                for each entry in X
        """
        X = self.X if X is None else X
        if hasattr(X, "values"):
            X = X.values
        N = X.shape[0]
        parms = []
        ids = []
        paths = []
        for i in range(N):
            path = []
            parms_i, id_i, path = self._find_params(X[i, :], path)
            parms.append(parms_i)
            ids.append(id_i)
            paths.append(path)
        return np.array(parms), np.array(ids), paths

    def predict(self, X=None):
        """Use fitted model to predict the response given X values.

        Args:
            X (array like [2D], optional): Independent variables to use
                when applying the tree. If None, the `X` used to fit the
                model is used. Defaults to None.

        Returns:
            np.array: vector or predicted response variables
        """
        X = self.X if X is None else X
        if hasattr(X, "values"):
            X = X.values
        params, ids, _ = self.apply(X)
        X = X[:, self.reg_vars]
        return (X * params).sum(axis=1)

    def _format_params_for_graph(self):
        min_width = 0
        for feat_id in self.reg_vars:
            width = len(self.feats[feat_id]) + 12
            if width > min_width:
                min_width = width

        param_format = [
            f"{i}: {j:5.3f}".rjust(min_width) for i, j in zip(self.feats, self.params)
        ]
        return param_format

    @staticmethod
    def _hex_to_rgb(value):
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def _rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb

    def _make_graphviz_labels(self, node, interps, nodelist, conlist):
        # get information about current node
        nid = node._ID
        mse = [node.mse]
        # interpolate for the color
        rgb = (int(interps[0](mse)[0]), int(interps[1](mse)[0]), int(interps[2](mse)[0]))
        myhex = self._rgb_to_hex(rgb)
        # determine text based on if it is a leaf or not
        if node.rule is None:
            param_format = node._format_params_for_graph()
            tag = "\n".join(param_format)
        else:
            tag = node.rule

        # formatting text
        mse_fmt = f"mse = {node.mse:.3f}\n"
        pct_smp = (
            f"samples = {node.N / PieceWiseLinearRegressionTree.init_N * 100:0.1f}%\n"
        )
        tag = mse_fmt + pct_smp + tag

        # append 'dot' information to nodelist
        state = f'"{nid}" [label="{tag}", fillcolor="{myhex}"]'
        nodelist.append(state)

        # check children (left and right)
        if node.left:
            child = node.left
            cid = child._ID
            if nid == 0:
                # when node is root, child is the first
                # less than or equal to split
                labelinfo = ["labeldistance=2.5", "labelangle=45", 'headlabel="True"']
                labelinfo = f"[{', '.join(labelinfo)}]"
                conlist.append(f'"{nid}" -> "{cid}" {labelinfo}')
            else:
                conlist.append(f'"{nid}" -> "{cid}"')
            # recurse down left child
            self._make_graphviz_labels(child, interps, nodelist, conlist)
        if node.right:
            child = node.right
            cid = child._ID
            if nid == 0:
                # when node is root, right child is first
                # greater than split
                labelinfo = ["labeldistance=2.5", "labelangle=-45", 'headlabel="False"']
                labelinfo = f"[{', '.join(labelinfo)}]"
                conlist.append(f'"{nid}" -> "{cid}" {labelinfo}')
            else:
                conlist.append(f'"{nid}" -> "{cid}"')
            # recurse down right child
            self._make_graphviz_labels(child, interps, nodelist, conlist)

    def to_graphviz(
        self, filename=None, shape="rectangle", graph="digraph", bgcolor="transparent"
    ):
        """Export tree in the ".dot" format for graphviz.

        Args:
            filename (str, optional): Filename to save to. If None, print to stdout.
            Defaults to None.
            shape (str, optional): shape of nodes. Defaults to "rectangle".
            graph (str, optional): graph style. Defaults to "digraph".
            bgcolor (str, optional): background color of graph. Defaults to "transparent".
        """
        # adapted from treelib.tree implementation
        """Exports the tree in the dot format of the graphviz software"""

        # coloring of nodes
        min_col = "#FFFFFF"
        max_col = "#E58139"
        min_rgb = self._hex_to_rgb(min_col)
        max_rgb = self._hex_to_rgb(max_col)

        # interpolate between max and min
        rinterp = interp1d(
            [
                PieceWiseLinearRegressionTree.min_mse,
                PieceWiseLinearRegressionTree.max_mse,
            ],
            [min_rgb[0], max_rgb[0]],
        )
        ginterp = interp1d(
            [
                PieceWiseLinearRegressionTree.min_mse,
                PieceWiseLinearRegressionTree.max_mse,
            ],
            [min_rgb[1], max_rgb[1]],
        )
        binterp = interp1d(
            [
                PieceWiseLinearRegressionTree.min_mse,
                PieceWiseLinearRegressionTree.max_mse,
            ],
            [min_rgb[2], max_rgb[2]],
        )
        interps = [rinterp, ginterp, binterp]

        # get information for graph output
        nodes, connections = [], []
        self._make_graphviz_labels(self, interps, nodes, connections)

        # write nodes and connections to dot format
        is_plain_file = filename is not None
        if is_plain_file:
            f = codecs.open(filename, "w", "utf-8")
        else:
            f = StringIO()

        # format for graph
        node_style = [
            f"shape={shape}",
            'style="filled, rounded"',
            'color="black"',
            "fontname=helvetica",
        ]
        edge_style = ["fontname=helvetica"]
        f.write(graph + " tree {\n")
        f.write(f'bgcolor="{bgcolor}"\n')
        f.write(f'node [{", ".join(node_style)}] ;\n')
        f.write(f'edge [{", ".join(edge_style)}] ;\n')

        for n in nodes:
            f.write("\t" + n + "\n")

        if len(connections) > 0:
            f.write("\n")

        for c in connections:
            f.write("\t" + c + "\n")

        f.write("}")

        if not is_plain_file:
            print(f.getvalue())

        f.close()

    def save_model(self, path, **kwargs):
        """Serialize model using pickle

        Args:
            path (str): Path to save serialized model to.
            kwargs (dict): Additional kwargs to pass to pickle.dump
        """
        if not kwargs.get("protocol"):
            kwargs["protocol"] = 4
        with open(path, "wb") as f:
            pickle.dump(self, f, **kwargs)
