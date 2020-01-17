import matplotlib.pyplot as plt
def printBoundaryDecision(pairidx):
  plot_colors = "ryb"  # Paramétro para el color de la figura
  plot_step = 0.02     # Parámetro para la figura
  # Imprimir el límite de decisión
  plt.subplot(2, 3, pairidx + 1)
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
  plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
  plt.xlabel(iris.feature_names[pair[0]])
  plt.ylabel(iris.feature_names[pair[1]])
  # Imprimir los puntos de entrenamiento
  for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

def showPlot():
  plt.suptitle("Decision surface of a decision tree using paired features")
  plt.legend(loc='lower right', borderpad=0, handletextpad=0)
  plt.axis("tight")

  plt.figure()
  clf = DecisionTreeClassifier().fit(iris.data, iris.target)
  plt.show()

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plotSVM():
  # title for the plots
  titles = ('SVC with linear kernel',
            'LinearSVC (linear kernel)',
            'SVC with RBF kernel',
            'SVC with polynomial (degree 3) kernel')

  # Set-up 2x2 grid for plotting.
  fig, sub = plt.subplots(2, 2)
  plt.subplots_adjust(wspace=0.4, hspace=0.4)

  X0, X1 = X[:, 0], X[:, 1]
  xx, yy = make_meshgrid(X0, X1)

  for clf, title, ax in zip(models, titles, sub.flatten()):
      plot_contours(ax, clf, xx, yy,
                    cmap=plt.cm.coolwarm, alpha=0.8)
      ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
      ax.set_xlim(xx.min(), xx.max())
      ax.set_ylim(yy.min(), yy.max())
      ax.set_xlabel('Sepal length')
      ax.set_ylabel('Sepal width')
      ax.set_xticks(())
      ax.set_yticks(())
      ax.set_title(title)
  plt.show()

def plotRegressorTree():
  # Plot the results
  plt.figure()
  plt.scatter(X, y, s=20, edgecolor="black",
              c="darkorange", label="data")
  plt.plot(X_test, y_1, color="cornflowerblue",
          label="max_depth=2", linewidth=2)
  plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
  plt.xlabel("data")
  plt.ylabel("target")
  plt.title("Decision Tree Regression")
  plt.legend()
  plt.show()

def plotSVR():
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
  for ix, svr in enumerate(svrs):
      axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                    label='{} model'.format(kernel_label[ix]))
      axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                      edgecolor=model_color[ix], s=50,
                      label='{} support vectors'.format(kernel_label[ix]))
      axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                      y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                      facecolor="none", edgecolor="k", s=50,
                      label='other training data')
      axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                      ncol=1, fancybox=True, shadow=True)

  fig.text(0.5, 0.04, 'data', ha='center', va='center')
  fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
  fig.suptitle("Support Vector Regression", fontsize=14)
  plt.show()
