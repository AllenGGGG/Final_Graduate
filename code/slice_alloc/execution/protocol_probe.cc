// Native NFD/ndnSIM forwarding and IPv4 on the SAME ns-3 NetDevices.
// A protocol integration probe, not a completed slicing/RL benchmark.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/ndnSIM-module.h"
#include "ns3/ndnSIM/apps/ndn-app.hpp"
#include "ns3/ndnSIM/model/ndn-app-link-service.hpp"
#include "ns3/ndnSIM/model/ndn-net-device-transport.hpp"
#include "ns3/ndnSIM/NFD/daemon/face/generic-link-service.hpp"
#include "ns3/ndnSIM/NFD/daemon/table/cs-policy.hpp"
#include "ns3/wifi-phy-state-helper.h"
#include "ns3/llc-snap-header.h"
#include "ns3/ndnSIM/model/ndn-block-header.hpp"
#include <list>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

using namespace ns3;
namespace {
std::ofstream events, frames, radioConfig;
bool usePeerUnicast=false, uniformL2=false, equalRetry=false;
std::string replayFile;
struct ReplayLease {std::string name;double start,end;};
std::vector<ReplayLease> replayLeases;
std::map<uint32_t,Ptr<NetDevice>> wirelessDevices;
std::map<std::pair<uint32_t,uint32_t>,std::shared_ptr<nfd::Face>> peerFaces;
uint64_t ipObjectRetries=0,ipObjectTimeouts=0;

uint64_t ndnSent=0, ndnReceived=0, ndnNacks=0, ipSent=0, ipReceived=0;
uint64_t ndnRetransmissions=0;
uint64_t deviceTx=0, objectIpSent=0, objectIpReceived=0;

uint64_t idSent=0,idReceived=0,idStaleData=0,idQueries=0,idRegistrations=0,idStaleRegistrations=0;
double Now() { return Simulator::Now().GetSeconds(); }

void Log(uint32_t node,const std::string& event,uint64_t face,
         const std::string& name,uint64_t bytes,double detail=0.) {
  events<<std::setprecision(12)<<Now()<<","<<node<<","<<event<<","<<face
        <<","<<name<<","<<bytes<<","<<detail<<"\n";
}
// Packet-level and PHY-state clocks are independent trace sources. State TX
// sums include control frames and preambles; not DIFS/backoff or channel idle.
void PhyBegin(uint32_t node,Ptr<const Packet> p,double) {
  Log(node,"phy_begin",p->GetUid(),"-",p->GetSize());
}
void PhyEnd(uint32_t node,Ptr<const Packet> p) {
  Log(node,"phy_end",p->GetUid(),"-",p->GetSize());
}
void PhyState(uint32_t node,Time start,Time duration,WifiPhyState state) {
  if(state==WifiPhyState::TX)Log(node,"phy_state_tx",0,"-",0,duration.GetSeconds());
}

void FrameTx(uint32_t node,uint32_t device,Ptr<const Packet> packet,uint16_t frequency,
             WifiTxVector vector,MpduInfo ampdu,uint16_t staId) {
  auto copy=packet->Copy();uint32_t mpduBytes=packet->GetSize();
  if(ampdu.type!=NORMAL_MPDU) {
    AmpduSubframeHeader delimiter;copy->RemoveHeader(delimiter);mpduBytes=delimiter.GetLength();
    NS_ABORT_MSG_IF(copy->GetSize()<mpduBytes,"truncated aggregate MPDU");
    copy->RemoveAtEnd(copy->GetSize()-mpduBytes); // discard per-subframe padding
  }
  WifiMacHeader header;copy->RemoveHeader(header);
  uint16_t protocol=0;
  if(header.IsData() && !header.IsQosAmsdu()) {LlcSnapHeader llc;if(copy->PeekHeader(llc))protocol=llc.GetType();}
  const auto mode=vector.GetMode(staId);
  std::string kind=header.IsData()?"data":(header.IsCtl()?"control":"management");
  std::string destination=header.GetAddr1().IsBroadcast()?"broadcast":(header.GetAddr1().IsGroup()?"multicast":"unicast");
  frames<<std::setprecision(12)<<Now()<<","<<node<<","<<device<<","<<packet->GetUid()<<","<<mpduBytes
    <<","<<kind<<","<<destination<<","<<header.GetAddr1()<<","<<header.GetAddr2()<<","<<protocol
    <<","<<mode.GetUniqueName()<<","<<mode.GetDataRate(vector,staId)<<","<<unsigned(vector.GetNss(staId))
    <<","<<vector.GetGuardInterval()<<","<<vector.GetChannelWidth()<<","<<unsigned(vector.GetPreambleType())
    <<","<<header.IsRetry()<<","<<vector.IsAggregation()<<","<<(ampdu.type==NORMAL_MPDU?0:ampdu.mpduRefNumber)<<","<<frequency
    <<","<<(header.IsQosData()?int(header.GetQosTid()):-1)<<","<<packet->GetSize()<<","<<(packet->GetSize()-mpduBytes)<<"\n";
}

// Explicit peer-addressed NFD transport. Multiple faces share ONE radio/device.
// Source/destination filtering is essential with promiscuous shared-medium RX.
class PeerUnicastTransport : public nfd::face::Transport {
public:
  PeerUnicastTransport(Ptr<Node> node,Ptr<NetDevice> device,Mac48Address peer)
    :m_node(node),m_device(device),m_peer(peer) {
    std::ostringstream local,remote;local<<"netdev://["<<Mac48Address::ConvertFrom(device->GetAddress())<<"]";
    remote<<"netdev://["<<peer<<"]";
    setLocalUri(nfd::FaceUri(local.str()));setRemoteUri(nfd::FaceUri(remote.str()));
    setScope(::ndn::nfd::FACE_SCOPE_NON_LOCAL);setPersistency(::ndn::nfd::FACE_PERSISTENCY_PERSISTENT);
    setLinkType(::ndn::nfd::LINK_TYPE_POINT_TO_POINT);setMtu(device->GetMtu());
    node->RegisterProtocolHandler(MakeCallback(&PeerUnicastTransport::Receive,this),
         ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE,device,true);
  }
  ~PeerUnicastTransport() override {
    m_node->UnregisterProtocolHandler(MakeCallback(&PeerUnicastTransport::Receive,this));
  }
private:
  void doClose() override {setState(nfd::face::TransportState::CLOSED);}
  void doSend(const ::ndn::Block& block) override {
    auto packet=Create<Packet>();packet->AddHeader(ns3::ndn::BlockHeader(block));
    Log(m_node->GetId(),"peer_tx",packet->GetUid(),"-",packet->GetSize());
    m_device->Send(packet,m_peer,ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE);
  }
  void Receive(Ptr<NetDevice> device,Ptr<const Packet> packet,uint16_t protocol,
               const Address& from,const Address& to,NetDevice::PacketType type) {
    if(device!=m_device || protocol!=ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE
       || type==NetDevice::PACKET_OTHERHOST || from!=m_peer || to!=m_device->GetAddress()) {
      Log(m_node->GetId(),"peer_filtered",packet->GetUid(),"-",packet->GetSize());return;
    }
    Log(m_node->GetId(),"peer_rx",packet->GetUid(),"-",packet->GetSize());
    auto copy=packet->Copy();ns3::ndn::BlockHeader header;copy->RemoveHeader(header);
    receive(std::move(header.getBlock()));
  }
  Ptr<Node> m_node;Ptr<NetDevice> m_device;Mac48Address m_peer;
};
void InjectWrongPeer() {
  ::ndn::Interest interest(::ndn::Name("/probe/filter"));interest.setNonce(123);
  auto packet=Create<Packet>();packet->AddHeader(ns3::ndn::BlockHeader(interest.wireEncode()));
  Log(2,"peer_negative_injection",packet->GetUid(),"/probe/filter",packet->GetSize());
  wirelessDevices.at(2)->Send(packet,wirelessDevices.at(3)->GetAddress(),ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE);
}

// Single-object pin proof, with byte accounting and a hard one-entry CS limit.
// Not yet a general multi-tenant per-slice state isolation implementation.
class LeasePinPolicy : public nfd::cs::Policy {
public:
  LeasePinPolicy():Policy("probe-lease-pin"){}
  bool pinned(EntryRef i) const {
    if(!replayFile.empty()) {
      for(const auto& lease:replayLeases)
        if(i->getName().toUri()==lease.name && Now()>=lease.start && Now()<lease.end)return true;
      return false;
    }
    return i->getName().toUri()=="/tenantA/slice1/map/v1/chunk0" && Now()>=1.05 && Now()<1.5;
  }
private:
  std::list<EntryRef> queue;
  void doAfterInsert(EntryRef i) override {
    if(!replayFile.empty() && !pinned(i)) {emitSignal(beforeEvict,i);return;}
    queue.push_back(i);evictEntries();
  }
  void doAfterRefresh(EntryRef) override {}
  void doBeforeUse(EntryRef) override {}
  void doBeforeErase(EntryRef i) override {queue.remove(i);}
  void evictEntries() override {
    while(getCs()->size()>getLimit()) {
      auto found=std::find_if(queue.begin(),queue.end(),[this](EntryRef i){return !pinned(i);});
      NS_ABORT_MSG_IF(found==queue.end(),"pin contract exceeds installed CS capacity");
      auto victim=*found;queue.erase(found);
      Log(1,"cs_evict",0,victim->getName().toUri(),victim->getData().getContent().value_size());
      emitSignal(beforeEvict,victim);
    }
  }
};
void CacheSnapshot(Ptr<Node> node) {
  auto& cs=ns3::ndn::L3Protocol::getL3Protocol(node)->getForwarder()->getCs();
  uint64_t payload=0,encoded=0;
  for(const auto& entry:cs) {
    payload+=entry.getData().getContent().value_size();encoded+=entry.getData().wireEncode().size();
    Log(node->GetId(),"cs_entry",0,entry.getName().toUri(),entry.getData().getContent().value_size(),entry.isFresh()?1.:0.);
  }
  Log(node->GetId(),"cs_payload",0,"-",payload,cs.size());
  Log(node->GetId(),"cs_encoded",0,"-",encoded,cs.size());
}
void ReleaseReplayLease(Ptr<Node> node,std::string name) {
  auto& cs=ns3::ndn::L3Protocol::getL3Protocol(node)->getForwarder()->getCs();
  cs.erase(::ndn::Name(name),100,[](size_t){});
  Log(node->GetId(),"replay_lease_release",0,name,0);
  CacheSnapshot(node);
}

// Equivalent-task UDP/IP unicast comparator: same names, versions, consumers,
// start times, payload bytes and deadlines. No app cache or multicast claimed.
class ObjectToken : public Header {
public:
  uint32_t seq=0;
  static TypeId GetTypeId() {static TypeId t=TypeId("ns3::ProtocolProbeObjectToken").SetParent<Header>().AddConstructor<ObjectToken>();return t;}
  TypeId GetInstanceTypeId() const override{return GetTypeId();}
  uint32_t GetSerializedSize() const override{return 4;}
  void Serialize(Buffer::Iterator i) const override{i.WriteHtonU32(seq);}
  uint32_t Deserialize(Buffer::Iterator i) override{seq=i.ReadNtohU32();return 4;}
  void Print(std::ostream& out) const override{out<<seq;}
};
NS_OBJECT_ENSURE_REGISTERED(ObjectToken);
struct ObjectPending {std::string name;double born;};
std::map<std::pair<uint32_t,uint32_t>,ObjectPending> objectPending;
uint32_t objectSequence=0;
void ObjectIpReceive(Ptr<Socket> socket) {
  Address from;
  while(auto p=socket->RecvFrom(from)) {
    ObjectToken token;if(equalRetry) {if(p->GetSize()<4)continue;p->RemoveHeader(token);}
    std::vector<uint8_t> bytes(p->GetSize());p->CopyData(bytes.data(),bytes.size());
    std::string name(bytes.begin(),bytes.end());
    if(socket->GetNode()->GetId()==0) {
      if(name.rfind("/tenant",0)!=0)continue;
      auto reply=Create<Packet>(1000);if(equalRetry)reply->AddHeader(token);
      socket->SendTo(reply,0,from);Log(0,"object_ip_produce",0,name,1000);
    } else {
      double delay=0.;
      if(equalRetry) {
        auto found=objectPending.find({socket->GetNode()->GetId(),token.seq});
        if(found==objectPending.end()){Log(socket->GetNode()->GetId(),"object_ip_duplicate",token.seq,"-",p->GetSize());continue;}
        name=found->second.name;delay=Now()-found->second.born;objectPending.erase(found);
      } else name="-";
      ++objectIpReceived;Log(socket->GetNode()->GetId(),"object_ip_data",0,name,p->GetSize(),delay);
    }
  }
}
void ObjectIpTransmit(Ptr<Socket> socket,uint32_t seq,bool retry) {
  const auto key=std::make_pair(socket->GetNode()->GetId(),seq);auto found=objectPending.find(key);
  if(found==objectPending.end())return;
  if(Now()>=found->second.born+5.) {++ipObjectTimeouts;Log(key.first,"object_ip_timeout",seq,found->second.name,0);objectPending.erase(found);return;}
  const auto& name=found->second.name;
  auto packet=Create<Packet>(reinterpret_cast<const uint8_t*>(name.data()),name.size());
  ObjectToken token;token.seq=seq;packet->AddHeader(token);socket->Send(packet);
  if(retry){++ipObjectRetries;Log(key.first,"object_ip_retransmit",seq,name,name.size());}
  Simulator::Schedule(Seconds(.113+.003*key.first),&ObjectIpTransmit,socket,seq,true);
}
void ObjectIpEmit(Ptr<Socket> socket,std::string name) {
  ++objectIpSent;Log(socket->GetNode()->GetId(),"object_ip_interest",0,name,name.size());
  if(equalRetry) {
    auto seq=++objectSequence;objectPending[{socket->GetNode()->GetId(),seq}]={name,Now()};
    ObjectIpTransmit(socket,seq,false);
  } else socket->Send(Create<Packet>(reinterpret_cast<const uint8_t*>(name.data()),name.size()));
}
void OutInterest(uint32_t node,const ::ndn::Interest& p,const nfd::Face& f) {
  Log(node,"out_interest",f.getId(),p.getName().toUri(),p.wireEncode().size());
}
void InInterest(uint32_t node,const ::ndn::Interest& p,const nfd::Face& f) {
  Log(node,"in_interest",f.getId(),p.getName().toUri(),p.wireEncode().size());
}
void OutData(uint32_t node,const ::ndn::Data& p,const nfd::Face& f) {
  Log(node,"out_data",f.getId(),p.getName().toUri(),p.wireEncode().size());
  if(node==0 && f.getId()==257)Log(node,"producer_payload",f.getId(),p.getName().toUri(),p.getContent().value_size());
}
void InData(uint32_t node,const ::ndn::Data& p,const nfd::Face& f) {
  Log(node,"in_data",f.getId(),p.getName().toUri(),p.wireEncode().size());
}
void DeviceTx(uint32_t node,uint32_t device,Ptr<const Packet> p) {
  ++deviceTx; Log(node,"device_tx",device,"-",p->GetSize());
}
void NativeWireRx(uint32_t node,Ptr<NetDevice>,Ptr<const Packet> p,uint16_t,
                  const Address&,const Address&,NetDevice::PacketType) {
  Log(node,"native_wire_rx",0,"-",p->GetSize());
}
std::shared_ptr<nfd::Face> RadioFace(Ptr<Node> node,Ptr<ns3::ndn::L3Protocol> l3,Ptr<NetDevice> device) {
  nfd::face::GenericLinkService::Options options;
  options.allowFragmentation=true;options.allowReassembly=true;
  if(usePeerUnicast) {
    std::shared_ptr<nfd::Face> first;
    for(uint32_t peer:{1,2,3}) {
      if(peer==node->GetId() || (peer!=1 && node->GetId()!=1))continue;
      auto transport=std::make_unique<PeerUnicastTransport>(node,device,Mac48Address::ConvertFrom(wirelessDevices.at(peer)->GetAddress()));
      auto face=std::make_shared<nfd::Face>(std::make_unique<nfd::face::GenericLinkService>(options),std::move(transport));
      face->setMetric(1);l3->addFace(face);peerFaces[{node->GetId(),peer}]=face;if(!first)first=face;
    }
    return first;
  }
  std::ostringstream local;local<<"netdev://["<<Mac48Address::ConvertFrom(device->GetAddress())<<"]";
  auto transport=std::make_unique<ns3::ndn::NetDeviceTransport>(node,device,local.str(),
      "netdev://[ff:ff:ff:ff:ff:ff]",::ndn::nfd::FACE_SCOPE_NON_LOCAL,
      ::ndn::nfd::FACE_PERSISTENCY_PERSISTENT,::ndn::nfd::LINK_TYPE_AD_HOC);
  auto face=std::make_shared<nfd::Face>(std::make_unique<nfd::face::GenericLinkService>(options),std::move(transport));
  face->setMetric(1);l3->addFace(face);return face;
}

class ProbeConsumer : public ns3::ndn::App {
public:
  static TypeId GetTypeId() {
    static TypeId t=TypeId("ns3::ProtocolProbeConsumer").SetParent<ns3::ndn::App>()
      .AddConstructor<ProbeConsumer>(); return t;
  }
  void Emit(std::string name) {
    pending[name].push_back(Now()); ++ndnSent;
    Transmit(name,Now(),0);
    Simulator::Schedule(Seconds(5.),&ProbeConsumer::Expire,this,name,Now());
  }
  void Transmit(std::string name,double born,uint32_t retry) {
    auto p=std::make_shared<::ndn::Interest>(::ndn::Name(name));
    p->setNonce(++nonce+1000*GetNode()->GetId());
    p->setInterestLifetime(::ndn::time::milliseconds(400));
    p->setMustBeFresh(true);
    if(retry)++ndnRetransmissions;
    Log(GetNode()->GetId(),retry?"app_retransmit":"app_interest",0,name,p->wireEncode().size());
    m_transmittedInterests(p,this,m_face);m_appLink->onReceiveInterest(*p);
    Simulator::Schedule(Seconds(.113+.003*GetNode()->GetId()),&ProbeConsumer::Retry,this,name,born,retry);
  }
  void Retry(std::string name,double born,uint32_t retry) {
    auto& q=pending[name];
    auto found=std::find(q.begin(),q.end(),born);
    if(found==q.end())return;
    if(Now()>=born+5.)return;
    Transmit(name,born,retry+1);
  }
  void Expire(std::string name,double born) {
    auto& q=pending[name];auto found=std::find(q.begin(),q.end(),born);
    if(found!=q.end()){Log(GetNode()->GetId(),"app_timeout",0,name,0,Now()-born);q.erase(found);}
  }
  void OnData(std::shared_ptr<const ::ndn::Data> p) override {
    ns3::ndn::App::OnData(p);
    auto& q=pending[p->getName().toUri()];
    if(q.empty()){Log(GetNode()->GetId(),"app_unsolicited_data",0,p->getName().toUri(),0);return;}
    ++ndnReceived;double delay=Now()-q.front();q.pop_front();
    Log(GetNode()->GetId(),"app_data",0,p->getName().toUri(),p->getContent().value_size(),delay);
  }
  void OnNack(std::shared_ptr<const ::ndn::lp::Nack> p) override {
    ns3::ndn::App::OnNack(p);++ndnNacks;
    Log(GetNode()->GetId(),"app_nack",0,p->getInterest().getName().toUri(),0);
    pending[p->getInterest().getName().toUri()].clear();
  }
private:
  uint32_t nonce=0;
  std::map<std::string,std::deque<double>> pending;
};
NS_OBJECT_ENSURE_REGISTERED(ProbeConsumer);

void ReceiveIp(Ptr<Socket> socket) {
  Address from;
  while(auto p=socket->RecvFrom(from)) {
    ++ipReceived;Log(socket->GetNode()->GetId(),"ip_receive",0,"-",p->GetSize());
  }
}
void SendIp(Ptr<Socket> socket) {
  if(Now()>5.4)return;
  int bytes=socket->Send(Create<Packet>(256));
  if(bytes>0)++ipSent;
  Log(socket->GetNode()->GetId(),"ip_send",0,"-",bytes>0?bytes:0);
  Simulator::Schedule(MilliSeconds(20),&SendIp,socket);
}

// Mechanism-level EID/RLOC probe. The 17-byte test header is deliberately NOT
// claimed as interoperable RFC 9300/9301 wire encoding or authentication.
class IdMessage : public Header {
public:
  uint8_t type=0;uint32_t eid=0x0aeeee01,version=0,locator=0,seq=0;
  static TypeId GetTypeId() {
    static TypeId t=TypeId("ns3::ProtocolProbeIdMessage").SetParent<Header>().AddConstructor<IdMessage>();return t;
  }
  TypeId GetInstanceTypeId() const override{return GetTypeId();}
  uint32_t GetSerializedSize() const override{return 17;}
  void Serialize(Buffer::Iterator b) const override {
    b.WriteU8(type);b.WriteHtonU32(eid);b.WriteHtonU32(version);b.WriteHtonU32(locator);b.WriteHtonU32(seq);
  }
  uint32_t Deserialize(Buffer::Iterator b) override {
    type=b.ReadU8();eid=b.ReadNtohU32();version=b.ReadNtohU32();locator=b.ReadNtohU32();seq=b.ReadNtohU32();return 17;
  }
  void Print(std::ostream& out) const override{out<<unsigned(type)<<":"<<eid<<":"<<version;}
};
NS_OBJECT_ENSURE_REGISTERED(IdMessage);
std::vector<Ptr<Socket>> idSockets;
Ptr<Socket> idSender,idServer;
Ipv4Address mapAddress;
uint32_t mapVersion=0,mapLocator=0,cachedVersion=0,cachedLocator=0,currentAttachment=2;
double cacheExpires=0.;bool queryPending=false;
std::deque<uint32_t> idPending;
void IdTransmit(uint32_t seq) {
  IdMessage h;h.type=4;h.version=cachedVersion;h.locator=cachedLocator;h.seq=seq;
  auto p=Create<Packet>(128);p->AddHeader(h);
  idSender->SendTo(p,0,InetSocketAddress(Ipv4Address(cachedLocator),4341));++idSent;
  Log(0,"id_encap",h.version,"eid/10.238.238.1",p->GetSize(),seq);
}
void IdReply(Ptr<Socket> socket) {
  Address from;
  while(auto p=socket->RecvFrom(from)) {
    if(p->GetSize()<17)continue;
    IdMessage h;p->RemoveHeader(h);
    if(h.type!=3||h.eid!=0x0aeeee01||h.version<cachedVersion)continue;
    cachedVersion=h.version;cachedLocator=h.locator;cacheExpires=Now()+.6;queryPending=false;
    Log(0,"id_map_reply",h.version,"eid/10.238.238.1",17,h.locator);
    while(!idPending.empty()){IdTransmit(idPending.front());idPending.pop_front();}
  }
}
void IdMapServer(Ptr<Socket> socket) {
  Address from;
  while(auto p=socket->RecvFrom(from)) {
    if(p->GetSize()<17)continue;
    IdMessage h;p->RemoveHeader(h);
    if(h.eid!=0x0aeeee01)continue;
    if(h.type==1) {
      if(h.version<mapVersion){++idStaleRegistrations;Log(1,"id_stale_register",h.version,"eid/10.238.238.1",17);continue;}
      mapVersion=h.version;mapLocator=h.locator;++idRegistrations;
      Log(1,"id_register",h.version,"eid/10.238.238.1",17,h.locator);
    } else if(h.type==2&&mapVersion) {
      Log(1,"id_query",mapVersion,"eid/10.238.238.1",17);
      h.type=3;h.version=mapVersion;h.locator=mapLocator;
      auto reply=Create<Packet>();reply->AddHeader(h);socket->SendTo(reply,0,from);
    }
  }
}
void IdEndpoint(Ptr<Socket> socket) {
  Address from;
  while(auto p=socket->RecvFrom(from)) {
    if(p->GetSize()<17)continue;
    IdMessage h;p->RemoveHeader(h);
    if(h.type!=4||h.eid!=0x0aeeee01)continue;
    auto node=socket->GetNode()->GetId();
    if(node!=currentAttachment) {++idStaleData;Log(node,"id_stale_data",h.version,"eid/10.238.238.1",p->GetSize(),h.seq);}
    else {++idReceived;Log(node,"id_decap",h.version,"eid/10.238.238.1",p->GetSize(),h.seq);}
  }
}
void IdRegister(Ptr<Socket> socket,Ipv4Address locator,uint32_t version) {
  IdMessage h;h.type=1;h.locator=locator.Get();h.version=version;
  auto p=Create<Packet>();p->AddHeader(h);socket->SendTo(p,0,InetSocketAddress(mapAddress,4342));
}
void IdMove(Ptr<Socket> socket,Ipv4Address locator) {
  currentAttachment=3;Log(3,"id_attachment_change",2,"eid/10.238.238.1",0);
  IdRegister(socket,locator,2);
}
void IdGenerate(uint32_t seq) {
  if(Now()>4.6)return;
  if(cacheExpires>Now()&&cachedLocator)IdTransmit(seq);
  else {
    idPending.push_back(seq);
    if(!queryPending) {
      queryPending=true;IdMessage h;h.type=2;
      auto p=Create<Packet>();p->AddHeader(h);idSender->SendTo(p,0,InetSocketAddress(mapAddress,4342));++idQueries;
    }
  }
  Simulator::Schedule(MilliSeconds(100),&IdGenerate,seq+1);
}
}

int main(int argc,char** argv) {
  std::string output="protocol-probe";
  bool cache=true,synchronous=true,wireless=false,ht=true,pin=false,pinControl=false,filterTest=false;
  std::string taskMode="coexist";
  double speed=0.,backhaulMbps=10.;uint32_t seed=1,cacheEntries=64;
  CommandLine cmd;cmd.AddValue("output","Output prefix",output);
  cmd.AddValue("cache","Enable relay content store",cache);
  cmd.AddValue("synchronous","Overlapping first Interests",synchronous);
  cmd.AddValue("wireless","Shared 802.11n last hop",wireless);
  cmd.AddValue("ht","Use 802.11n (otherwise 802.11a diagnostic)",ht);
  cmd.AddValue("speed","Consumer b horizontal speed m/s",speed);
  cmd.AddValue("pinControl","Run same pressure tasks under ordinary LRU",pinControl);
  cmd.AddValue("pin","Test one-object pinned cache eviction enforcement",pin);
  cmd.AddValue("taskMode","coexist, equivalent_ndn or equivalent_ip",taskMode);
  cmd.AddValue("peerUnicast","Use explicit per-neighbor NDN unicast transport",usePeerUnicast);
  cmd.AddValue("uniformL2","ERP6 data/control/group mode, no aggregation, prepared neighbors",uniformL2);
  cmd.AddValue("equalRetry","IP object retry/deadline contract matched to NDN",equalRetry);
  cmd.AddValue("filterTest","Inject packet from a non-peer before workload",filterTest);
  cmd.AddValue("replayFile","TSV: time node ip|ndn name lease_start lease_end",replayFile);
  cmd.AddValue("backhaulMbps","Explicit physical producer-relay rate",backhaulMbps);
  cmd.AddValue("cacheEntries","Payload object slots for replay leases",cacheEntries);
  cmd.AddValue("seed","Independent ns-3 RNG seed",seed);cmd.Parse(argc,argv);
  NS_ABORT_MSG_IF(backhaulMbps<=0 || cacheEntries==0,"invalid replay resource capacity");
  RngSeedManager::SetSeed(seed);RngSeedManager::SetRun(1);
  radioConfig.open(output+"-radio-config.csv");
  radioConfig<<"node,device,manager,data_mode,control_mode,nonunicast_mode,requested_max_ssrc,requested_max_slrc,BE_ampdu,BE_amsdu,BK_ampdu,BK_amsdu,VI_ampdu,VI_amsdu,VO_ampdu,VO_amsdu\n";
  frames.open(output+"-frames.csv");frames<<"time,node,device,uid,bytes,frame_type,destination,addr1,addr2,ethertype,mode,phy_bps,nss,gi_ns,width_mhz,preamble,retry,aggregate,ampdu_ref,frequency_mhz,qos_tid,monitor_bytes,aggregate_overhead_bytes\n";
  events.open(output+"-events.csv");events<<"time,node,event,face,name,bytes,detail\n";
  NodeContainer nodes;nodes.Create(4); // producer, relay/AP, consumer a, consumer b
  MobilityHelper mobility;mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
  mobility.Install(nodes);
  const Vector positions[]={Vector(-10,0,0),Vector(0,0,10),Vector(20,0,80),Vector(30,0,80)};
  for(uint32_t i=0;i<4;++i)nodes.Get(i)->GetObject<MobilityModel>()->SetPosition(positions[i]);
  nodes.Get(3)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(Vector(speed,0,0));
  PointToPointHelper p2p;p2p.SetDeviceAttribute("DataRate",DataRateValue(DataRate(uint64_t(backhaulMbps*1e6))));
  p2p.SetChannelAttribute("Delay",StringValue("10ms"));
  auto upstream=p2p.Install(nodes.Get(0),nodes.Get(1));
  NetDeviceContainer accessA,accessB,radio;
  if(wireless) {
    NodeContainer access;access.Add(nodes.Get(1));access.Add(nodes.Get(2));access.Add(nodes.Get(3));
    YansWifiChannelHelper channel;channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::RangePropagationLossModel","MaxRange",DoubleValue(120.));
    YansWifiPhyHelper phy;phy.SetChannel(channel.Create());
    WifiHelper wifi;wifi.SetStandard(ht?WIFI_STANDARD_80211n_2_4GHZ:WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue(uniformL2?"ErpOfdmRate6Mbps":(ht?"HtMcs0":"OfdmRate6Mbps")),
                                 "ControlMode",StringValue(ht?"ErpOfdmRate6Mbps":"OfdmRate6Mbps"));
    if(uniformL2)wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue("ErpOfdmRate6Mbps"),
        "ControlMode",StringValue("ErpOfdmRate6Mbps"),"NonUnicastMode",StringValue("ErpOfdmRate6Mbps"),
        "MaxSsrc",UintegerValue(7),"MaxSlrc",UintegerValue(7));
    WifiMacHelper mac;mac.SetType("ns3::AdhocWifiMac");radio=wifi.Install(phy,mac,access);
    for(uint32_t j=0;j<3;++j) {
      wirelessDevices[j+1]=radio.Get(j);
      if(uniformL2)for(std::string ac:{"BE","BK","VI","VO"}) {
        auto mac=DynamicCast<WifiNetDevice>(radio.Get(j))->GetMac();
        mac->SetAttribute(ac+"_MaxAmpduSize",UintegerValue(0));mac->SetAttribute(ac+"_MaxAmsduSize",UintegerValue(0));
      }
    }
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    phy.EnablePcap(output,radio,true);
  } else {
    accessA=p2p.Install(nodes.Get(1),nodes.Get(2));
    accessB=p2p.Install(nodes.Get(1),nodes.Get(3));
  }
  InternetStackHelper internet;internet.Install(nodes);
  Ipv4AddressHelper address;address.SetBase("10.0.0.0","255.255.255.0");
  auto producerAddresses=address.Assign(upstream);
  address.SetBase("10.0.1.0","255.255.255.0");
  Ipv4Address locatorA,locatorB;
  if(wireless) {
    auto interfaces=address.Assign(radio);locatorA=interfaces.GetAddress(1);locatorB=interfaces.GetAddress(2);
  } else {
    auto a=address.Assign(accessA);locatorA=a.GetAddress(1);
    address.SetBase("10.0.2.0","255.255.255.0");auto b=address.Assign(accessB);locatorB=b.GetAddress(1);
  }
  Ipv4GlobalRoutingHelper::PopulateRoutingTables();
  if(uniformL2 && wireless)for(auto& local:wirelessDevices) {
    auto ipv4=local.second->GetNode()->GetObject<Ipv4L3Protocol>();
    auto cache=ipv4->GetInterface(ipv4->GetInterfaceForDevice(local.second))->GetArpCache();
    for(auto& peer:wirelessDevices)if(peer.first!=local.first) {
      auto other=peer.second->GetNode()->GetObject<Ipv4>();
      auto ip=other->GetAddress(other->GetInterfaceForDevice(peer.second),0).GetLocal();
      auto entry=cache->Add(ip);entry->MarkAlive(peer.second->GetAddress());entry->MarkPermanent();
    }
    Log(local.first,"neighbor_prepared",local.second->GetIfIndex(),"all-radio-peers",0);
  }
  if(filterTest)Simulator::Schedule(Seconds(.15),&InjectWrongPeer);
  ns3::ndn::StackHelper stack;
  stack.AddFaceCreateCallback(WifiNetDevice::GetTypeId(),MakeCallback(&RadioFace));
  stack.setCsSize(0);stack.Install(nodes.Get(0));
  stack.Install(nodes.Get(2));stack.Install(nodes.Get(3));
  stack.setCsSize(cache?(pin?1:cacheEntries):0);stack.Install(nodes.Get(1));
  if((pin && !pinControl) || !replayFile.empty()) {
    auto& cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(1))->getForwarder()->getCs();
    cs.setPolicy(std::make_unique<LeasePinPolicy>());cs.setLimit(replayFile.empty()?1:cacheEntries);
  }
  for(double at:{1.10,1.25,1.45,1.75,2.15,2.45,2.75})
    Simulator::Schedule(Seconds(at),&CacheSnapshot,nodes.Get(1));
  for(const std::string prefix:{"/tenantA","/tenantB"}) {
    ns3::ndn::StrategyChoiceHelper::InstallAll(prefix,"/localhost/nfd/strategy/best-route");
    ns3::ndn::FibHelper::AddRoute(nodes.Get(1),prefix,nodes.Get(0),1);
    if(wireless) {
      ns3::ndn::FibHelper::AddRoute(nodes.Get(2),prefix,usePeerUnicast?peerFaces.at({2,1}):ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(2))->getFaceByNetDevice(radio.Get(1)),1);
      ns3::ndn::FibHelper::AddRoute(nodes.Get(3),prefix,usePeerUnicast?peerFaces.at({3,1}):ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(3))->getFaceByNetDevice(radio.Get(2)),1);
    } else {
      ns3::ndn::FibHelper::AddRoute(nodes.Get(2),prefix,nodes.Get(1),1);
      ns3::ndn::FibHelper::AddRoute(nodes.Get(3),prefix,nodes.Get(1),1);
    }
    ns3::ndn::AppHelper producer("ns3::ndn::Producer");producer.SetPrefix(prefix);
    producer.SetAttribute("PayloadSize",UintegerValue(1000));
    producer.SetAttribute("Freshness",TimeValue(replayFile.empty()?MilliSeconds(500):Seconds(60)));
    producer.Install(nodes.Get(0));
  }
  auto ca=CreateObject<ProbeConsumer>(),cb=CreateObject<ProbeConsumer>();
  nodes.Get(2)->AddApplication(ca);nodes.Get(3)->AddApplication(cb);
  ca->SetStartTime(Seconds(.1));cb->SetStartTime(Seconds(.1));
  const std::string object="/tenantA/slice1/map/v1/chunk0";
  std::vector<std::tuple<double,uint32_t,std::string>> tasks={
    {1.,2,object},{synchronous?1.001:1.2,3,object},{1.3,2,object},{2.,2,object},
    {2.3,2,"/tenantA/slice1/map/v2/chunk0"},{2.6,3,"/tenantB/slice2/map/v1/chunk0"}};
  if(pin) {
    tasks.push_back({1.15,2,"/tenantA/slice1/pressure/v1/chunk0"});
    tasks.push_back({1.7,2,"/tenantA/slice1/pressure/v2/chunk0"});
  }
  std::map<std::tuple<double,uint32_t,std::string>,std::string> replayModes;
  double stopAt=9.;
  if(!replayFile.empty()) {
    tasks.clear();std::ifstream input(replayFile);NS_ABORT_MSG_IF(!input,"cannot open replay task file");
    std::string line;std::set<std::string> leased;
    while(std::getline(input,line)) {
      if(line.empty() || line[0]=='#')continue;
      std::istringstream fields(line);double at,start,end;uint32_t node;std::string mode,name;
      fields>>at>>node>>mode>>name>>start>>end;
      NS_ABORT_MSG_IF(!fields || at<.1 || (node!=2 && node!=3) || (mode!="ip" && mode!="ndn")
          || name.rfind("/tenant",0)!=0 || start<0 || end<start,"invalid replay row");
      auto key=std::make_tuple(at,node,name);
      NS_ABORT_MSG_IF(replayModes.count(key),"duplicate replay task");
      tasks.push_back(key);replayModes[key]=mode;stopAt=std::max(stopAt,at+5.1);
      if(end>start && leased.insert(name).second) {
        NS_ABORT_MSG_IF(mode!="ndn" || start>at || at>=end,"invalid replay pin interval");
        replayLeases.push_back({name,start,end});
        Simulator::Schedule(Seconds(end),&ReleaseReplayLease,nodes.Get(1),name);
      }
    }
    taskMode="replay";
  }
  NS_ABORT_MSG_IF(taskMode!="coexist" && taskMode!="equivalent_ndn" && taskMode!="equivalent_ip" && taskMode!="replay","unknown task mode");
  if(taskMode=="equivalent_ip" || taskMode=="replay") {
    auto server=Socket::CreateSocket(nodes.Get(0),UdpSocketFactory::GetTypeId());
    server->Bind(InetSocketAddress(Ipv4Address::GetAny(),9100));server->SetRecvCallback(MakeCallback(&ObjectIpReceive));
    std::map<uint32_t,Ptr<Socket>> clients;
    for(uint32_t i:{2,3}) {
      auto c=Socket::CreateSocket(nodes.Get(i),UdpSocketFactory::GetTypeId());c->Bind();
      c->Connect(InetSocketAddress(producerAddresses.GetAddress(0),9100));
      c->SetRecvCallback(MakeCallback(&ObjectIpReceive));clients[i]=c;
    }
    for(auto& task:tasks) {
      if(taskMode=="replay" && replayModes.at(task)=="ndn")
        Simulator::Schedule(Seconds(std::get<0>(task)),&ProbeConsumer::Emit,std::get<1>(task)==2?ca:cb,std::get<2>(task));
      else Simulator::Schedule(Seconds(std::get<0>(task)),&ObjectIpEmit,clients[std::get<1>(task)],std::get<2>(task));
    }
  } else {
    for(auto& task:tasks)Simulator::Schedule(Seconds(std::get<0>(task)),&ProbeConsumer::Emit,std::get<1>(task)==2?ca:cb,std::get<2>(task));
    if(taskMode=="coexist")Simulator::Schedule(Seconds(3.),&ProbeConsumer::Emit,ca,std::string("/unknown/map/v1/chunk0"));
  }

  uint32_t sharedDevices=0;
  for(uint32_t i=0;i<nodes.GetN();++i) {
    nodes.Get(i)->RegisterProtocolHandler(MakeBoundCallback(&NativeWireRx,i),
      ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE,Ptr<NetDevice>(),true);
    auto l3=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(i));
    l3->TraceConnectWithoutContext("OutInterests",MakeBoundCallback(&OutInterest,i));
    l3->TraceConnectWithoutContext("InInterests",MakeBoundCallback(&InInterest,i));
    l3->TraceConnectWithoutContext("OutData",MakeBoundCallback(&OutData,i));
    l3->TraceConnectWithoutContext("InData",MakeBoundCallback(&InData,i));
    for(uint32_t d=0;d<nodes.Get(i)->GetNDevices();++d) {
      auto device=nodes.Get(i)->GetDevice(d);
      if(device->GetInstanceTypeId().GetName()=="ns3::LoopbackNetDevice")continue;
      auto face=l3->getFaceByNetDevice(device);
      if(!face && usePeerUnicast && DynamicCast<WifiNetDevice>(device))face=peerFaces.at({i,i==1?2:1});
      if(face) {
        NS_ABORT_MSG_IF(nodes.Get(i)->GetObject<Ipv4>()->GetInterfaceForDevice(device)<0,
                        "NDN device lacks shared IPv4 interface");
        ++sharedDevices;
        Log(i,"device_face",face->getId(),"device/"+std::to_string(d),0);
        device->TraceConnectWithoutContext("MacTx",MakeBoundCallback(&DeviceTx,i,d));
        if(auto wifi=DynamicCast<WifiNetDevice>(device)) {
          auto manager=wifi->GetRemoteStationManager();WifiModeValue data,control,nonunicast;UintegerValue ssrc(7),slrc(uniformL2?7:4);
          manager->GetAttribute("DataMode",data);manager->GetAttribute("ControlMode",control);manager->GetAttribute("NonUnicastMode",nonunicast);
          // ns-3.35 exposes setter-only attributes for retry limits. Record
          // requested/default values honestly, not a claimed getter readback.
          radioConfig<<i<<","<<d<<","<<manager->GetInstanceTypeId().GetName()<<","<<data.Get().GetUniqueName()<<","<<control.Get().GetUniqueName()
            <<","<<(nonunicast.Get()==WifiMode()?"default":nonunicast.Get().GetUniqueName())<<","<<ssrc.Get()<<","<<slrc.Get();
          for(std::string ac:{"BE","BK","VI","VO"}) {
            UintegerValue ampdu,amsdu;wifi->GetMac()->GetAttribute(ac+"_MaxAmpduSize",ampdu);wifi->GetMac()->GetAttribute(ac+"_MaxAmsduSize",amsdu);
            radioConfig<<","<<ampdu.Get()<<","<<amsdu.Get();
          }
          radioConfig<<"\n";
          wifi->GetMac()->TraceConnectWithoutContext("MacTx",MakeBoundCallback(&DeviceTx,i,d));
          wifi->GetPhy()->TraceConnectWithoutContext("MonitorSnifferTx",MakeBoundCallback(&FrameTx,i,d));
          wifi->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeBoundCallback(&PhyBegin,i));
          wifi->GetPhy()->TraceConnectWithoutContext("PhyTxEnd",MakeBoundCallback(&PhyEnd,i));
          wifi->GetPhy()->GetState()->TraceConnectWithoutContext("State",MakeBoundCallback(&PhyState,i));
        }
      }
    }
  }
  if(taskMode=="coexist") {
  auto sink=Socket::CreateSocket(nodes.Get(0),UdpSocketFactory::GetTypeId());
  sink->Bind(InetSocketAddress(Ipv4Address::GetAny(),9000));sink->SetRecvCallback(MakeCallback(&ReceiveIp));
  for(uint32_t i:{2,3}) {
    auto source=Socket::CreateSocket(nodes.Get(i),UdpSocketFactory::GetTypeId());
    source->Connect(InetSocketAddress(producerAddresses.GetAddress(0),9000));
    Simulator::Schedule(Seconds(.4),&SendIp,source);
  }
  mapAddress=producerAddresses.GetAddress(1);
  idServer=Socket::CreateSocket(nodes.Get(1),UdpSocketFactory::GetTypeId());
  idServer->Bind(InetSocketAddress(Ipv4Address::GetAny(),4342));idServer->SetRecvCallback(MakeCallback(&IdMapServer));
  idSender=Socket::CreateSocket(nodes.Get(0),UdpSocketFactory::GetTypeId());
  idSender->Bind();idSender->SetRecvCallback(MakeCallback(&IdReply));
  for(uint32_t i:{2,3}) {
    auto endpoint=Socket::CreateSocket(nodes.Get(i),UdpSocketFactory::GetTypeId());
    endpoint->Bind(InetSocketAddress(Ipv4Address::GetAny(),4341));endpoint->SetRecvCallback(MakeCallback(&IdEndpoint));
    idSockets.push_back(endpoint);
  }
  Simulator::Schedule(Seconds(.2),&IdRegister,idSockets[0],locatorA,1);
  Simulator::Schedule(Seconds(.6),&IdGenerate,0);
  Simulator::Schedule(Seconds(2.5),&IdMove,idSockets[1],locatorB);
  Simulator::Schedule(Seconds(2.8),&IdRegister,idSockets[0],locatorA,0); // stale update must not win
  }
  // Last request at t=3 plus its 5 s contract; do not censor pending requests.
  Simulator::Stop(Seconds(stopAt));Simulator::Run();
  auto final=nodes.Get(3)->GetObject<MobilityModel>()->GetPosition();
  std::ofstream summary(output+"-summary.json");
  summary<<"{\"native_ndnsim\":true,\"wireless\":"<<(wireless?"true":"false")
    <<",\"cache\":"<<(cache?"true":"false")<<",\"synchronous\":"<<(synchronous?"true":"false")
    <<",\"pin\":"<<(pin?"true":"false")<<",\"task_mode\":\""<<taskMode<<"\""
    <<",\"peer_unicast\":"<<(usePeerUnicast?"true":"false")<<",\"uniform_l2\":"<<(uniformL2?"true":"false")
    <<",\"equal_retry\":"<<(equalRetry?"true":"false")<<",\"object_ip_retransmissions\":"<<ipObjectRetries<<",\"object_ip_timeouts\":"<<ipObjectTimeouts
    <<",\"object_ip_sent\":"<<objectIpSent<<",\"object_ip_received\":"<<objectIpReceived
    <<",\"speed_mps\":"<<speed<<",\"seed\":"<<seed<<",\"shared_devices\":"<<sharedDevices
    <<",\"ndn_sent\":"<<ndnSent<<",\"ndn_received\":"<<ndnReceived<<",\"ndn_nacks\":"<<ndnNacks
    <<",\"ndn_retransmissions\":"<<ndnRetransmissions
    <<",\"ip_sent\":"<<ipSent<<",\"ip_received\":"<<ipReceived<<",\"device_tx\":"<<deviceTx
    <<",\"id_sent\":"<<idSent<<",\"id_received\":"<<idReceived<<",\"id_stale_data\":"<<idStaleData
    <<",\"id_queries\":"<<idQueries<<",\"id_registrations\":"<<idRegistrations
    <<",\"id_stale_registrations\":"<<idStaleRegistrations<<",\"id_final_mapping_version\":"<<mapVersion
    <<",\"consumer_b_final_x\":"<<final.x<<"}\n";
  Simulator::Destroy();return 0;
}
