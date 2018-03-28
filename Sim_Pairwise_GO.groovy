@Grab(group='com.github.sharispe', module='slib-sml', version='0.9.1')
@Grab(group='org.codehaus.gpars', module='gpars', version='1.1.0')

import java.net.*
import org.openrdf.model.vocabulary.*
import slib.sglib.io.loader.*
import slib.sml.sm.core.metrics.ic.utils.*
import slib.sml.sm.core.utils.*
import slib.sglib.io.loader.bio.obo.*
import org.openrdf.model.URI
import slib.graph.algo.extraction.rvf.instances.*
import slib.sglib.algo.graph.utils.*
import slib.utils.impl.Timer
import slib.graph.algo.extraction.utils.*
import slib.graph.model.graph.*
import slib.graph.model.repo.*
import slib.graph.model.impl.graph.memory.*
import slib.sml.sm.core.engine.*
import slib.graph.io.conf.*
import slib.graph.model.impl.graph.elements.*
import slib.graph.algo.extraction.rvf.instances.impl.*
import slib.graph.model.impl.repo.*
import slib.graph.io.util.*
import slib.graph.io.loader.*
import groovyx.gpars.GParsPool



def factory = URIFactoryMemory.getSingleton()

class Gene {

  int id
  Set annotations

  public Gene(id, annotations) {
    setId(id)
    setAnnotations(annotations)
  }

  void addAnnotation(annotation) {
    annotations.add(annotation);
  }

  def getAnnotations() {
    annotations
  }

}


def getGeneOntology = {

  URI graph_uri = factory.getURI("http://purl.obolibrary.org/obo/")
  factory.loadNamespacePrefix("GO", graph_uri.toString())
  G graph = new GraphMemory(graph_uri)

  // Load OBO file to graph "go.obo"
  GDataConf goConf = new GDataConf(GFormat.RDF_XML, "data/ontologies1/go.owl")
  GraphLoaderGeneric.populate(goConf, graph)

  // Add virtual root for 3 subontologies__________________________________
  URI virtualRoot = factory.getURI("http://purl.obolibrary.org/obo/virtualRoot")
  graph.addV(virtualRoot)
  GAction rooting = new GAction(GActionType.REROOTING)
  rooting.addParameter("root_uri", virtualRoot.stringValue())
  GraphActionExecutor.applyAction(factory, rooting, graph)
  return graph
}

def getURIfromGO = { go ->
  // def id = go.split('_')[1]
  return factory.getURI("http://purl.obolibrary.org/obo/" + go)
}

def getGenes = {
  def genes = []
  def i = 0
  new File("data/ontologies1/gene_go_annotation.tab").splitEachLine('\t') { items ->
    def s = 0
    if (!items[0].startsWith("#")) {
      genes.push(new Gene(i, new LinkedHashSet()))
      // def gos = items[2].split("; ")
      for (int j = 1; j < items.size(); j++) {
        genes[i].addAnnotation(getURIfromGO(items[j]))
      }
      if (genes[i].getAnnotations().size() == 0) {
        println(items[0] + " " + i)
      }
      i++
    }
  }
  return genes
}

graph = getGeneOntology()
genes = getGenes()

def sim_id = 0 // this.args[0].toInteger()

SM_Engine engine = new SM_Engine(graph)

// BMA+Resnik, BMA+Schlicker2006, BMA+Lin1998, BMA+Jiang+Conrath1997,
// DAG-GIC, DAG-NTO, DAG-UI

String[] flags = [
  // SMConstants.FLAG_SIM_GROUPWISE_AVERAGE
  // SMConstants.FLAG_SIM_GROUPWISE_AVERAGE_NORMALIZED_GOSIM,
  SMConstants.FLAG_SIM_GROUPWISE_BMA
  // SMConstants.FLAG_SIM_GROUPWISE_BMM,
  // SMConstants.FLAG_SIM_GROUPWISE_MAX,
  // SMConstants.FLAG_SIM_GROUPWISE_MIN,
  // SMConstants.FLAG_SIM_GROUPWISE_MAX_NORMALIZED_GOSIM
]

// List<String> pairFlags = new ArrayList<String>(SMConstants.PAIRWISE_MEASURE_FLAGS);
String[] pairFlags = [
  SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995
  // SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_SCHLICKER_2006,
  // SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_LIN_1998,
  // SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_JIANG_CONRATH_1997_NORM
]

ICconf icConf = new IC_Conf_Topo("Sanchez", SMConstants.FLAG_ICI_SANCHEZ_2011);
String flagGroupwise = flags[sim_id.intdiv(pairFlags.size())];
String flagPairwise = pairFlags[sim_id % pairFlags.size()];
SMconf smConfGroupwise = new SMconf(flagGroupwise);
SMconf smConfPairwise = new SMconf(flagPairwise);
smConfPairwise.setICconf(icConf);

// Schlicker indirect
ICconf prob = new IC_Conf_Topo(SMConstants.FLAG_ICI_PROB_OCCURENCE_PROPAGATED);
smConfPairwise.addParam("ic_prob", prob);

def result = new Double[genes.size() * genes.size()]
for (i = 0; i < result.size(); i++) {
  result[i] = i
}

def c = 0

GParsPool.withPool {
  result.eachParallel { val ->
    def i = val.toInteger()
    def x = i.intdiv(genes.size())
    def y = i % genes.size()
    if (x <= y) {
      result[i] = engine.compare(
              smConfGroupwise,
              smConfPairwise,
              genes[x].getAnnotations(),
              genes[y].getAnnotations())
      if (c % 100000 == 0)
        println c
      c++
    }
  }
}

def fout = new PrintWriter(new BufferedWriter(
  new FileWriter("data/ontologies1/go.txt")))
for (i = 0; i < result.size(); i++) {
  def x = i.intdiv(genes.size())
  def y = i % genes.size()
  if (x <= y) {
    fout.println(result[i])
  } else {
    def j = y * genes.size() + x
    fout.println(result[j])
  }
}
fout.flush()
fout.close()
